import sys

import torch.utils
import torch.utils.data
sys.path.append('../')
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/mnt/data/ckpt/linzhisheng/cache'
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from model.modeling_qwen2_moe import Qwen2MoeForCausalLM
from utils.common_utils import read_json, write_json
import torch
import json
import sqlglot
from sqlglot.optimizer import optimize
from sqlglot import parse_one, diff
from sqlglot.diff import Keep
from utils.value_match import result_eq
from utils.df_eval import subset_df
from utils.prompt_utils import gen_train_prompt
from tqdm import tqdm
from database.PG import PG
from database.Mysql import MySQL
from database.SQLite import SQLite
import pandas as pd
from accelerate import Accelerator
from peft import PeftModel
from accelerate import infer_auto_device_map
import re
accelerator = Accelerator()


def extract_sql_from_qwen(qwen_result) -> str:
    sql = qwen_result
    pattern = r"```sql(.*?)```"

    # 使用re.DOTALL标志来使得点号(.)可以匹配包括换行符在内的任意字符
    sql_code_snippets = re.findall(pattern, qwen_result, re.DOTALL)

    if len(sql_code_snippets) > 0:
        sql = sql_code_snippets[-1].strip()

    return sql

def contains_none(res: list[tuple])->bool:
    if res is None:
        return True
    if len(res) == 0:
        return True
    else:
        for res_tuple in res:# for each tuple
            if len(res_tuple) > 0 and None in res_tuple:
                return True
    return False

class Evaluator():

    def __init__(self, model_path, dialect, eval_data_path, save_path, device='auto'):
        
        self.sql_dialect = dialect
        self.model_path = model_path
        self.eval_data_path = eval_data_path
        self.save_path = save_path

        self.batch_size = 4
        self.device = device
    def inference(self):
        
        config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        config.use_cache=True
        print(config)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        tokenizer.padding_side = "left"
        device = self.device
        if self.model_path.find('moe')>=0:
            print('loading from Qwen2MoeForCausalLM')
            model = Qwen2MoeForCausalLM.from_pretrained(
                self.model_path,
                config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                device_map=device
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
                device_map=device
            )
            # peft_model_id = "output/dense/0624/qwen2_32b_lora_no_shuffle_1e-6_question/checkpoint-500"
            # model = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map='auto')
            # model = model.merge_and_unload()
        # model = torch.compile(model)
        eval_json = read_json(self.eval_data_path)
        print(len(eval_json))

        def chunk_data(data, chunk_size):
            for i in range(0, len(data), chunk_size):
                yield data[i : min(i + chunk_size, len(data))]
        data_chunks = list(chunk_data(eval_json, self.batch_size))

        final_result = []
        print(len(data_chunks))
        for idx, t in tqdm(enumerate(data_chunks)):
            with accelerator.split_between_processes(t,) as batch:
                texts = []
                for row in batch:
                    # prompt = gen_train_prompt(idx, row, self.sql_dialect)
                    # conversations = prompt['conversations'][:1]
                    conversations = row['conversations'][:1]
                    text = tokenizer.apply_chat_template(
                        conversations,
                        tokenize=False,
                        # chat_template=TEMPLATE,
                        add_generation_prompt=True
                    )
                    if idx == 0:
                        print(text)
                    texts.append(text)

                model_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False).to(accelerator.device)
                generated_ids = model.generate(
                    **model_inputs,
                    pad_token_id = tokenizer.pad_token_id,
                    eos_token_id = tokenizer.eos_token_id,
                    max_new_tokens=512,
                    temperature=None,
                    num_beams=1,
                    top_p=None,
                    do_sample=False,
                )
                torch.cuda.empty_cache()
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                for i in range(len(response)):
                    batch[i]['pred_sql'] = response[i]
                    batch[i]['sql'] = batch[i]['conversations'][1]['content']
                    final_result.append(batch[i])
                    # eval_json[idx*self.batch_size+i]['pred_sql'] = response[i]
                # print(target)
                print(response)
                # if idx % 5 == 0:
                    # write_json(self.model_path + 'result.json', eval_json)

        write_json(self.save_path + f'/result_{accelerator.process_index}.json', final_result)

    def sql_compare(self, sql_gt, sql_pred, dialect):
        
        dialect=dialect.lower()
        if dialect=='postgresql':
            dialect = 'postgres'
        try:
            sql_gt = sqlglot.transpile(sql_gt, read=dialect,
                                    write=dialect, identify=True, pretty=False)[0]
            sql_pred = \
                sqlglot.transpile(sql_pred, read=dialect,
                                write=dialect, identify=True, pretty=False)[0]
        except:
            pass

        try:
            optimized_source = optimize(parse_one(sql_gt, dialect=dialect))
            optimized_pred = optimize(parse_one(sql_pred, dialect=dialect))
            edit_script = diff(optimized_source, optimized_pred)
            _ = sum(0 if isinstance(e, Keep) else 1 for e in edit_script)
            if _ == 0:
                exact_score = True
            else:
                exact_score = False
        except:
            exact_score = False

        return exact_score
    
    def evaluate(self):
        # self.result_data = read_json(self.model_path + '/result.json')
        self.result_data = []
        # for i in range(4):
        #     self.result_data.extend(read_json(self.save_path + f'/result_{i}.json'))
        # print(len(self.result_data))
        self.result_data.extend(read_json('data/lzs_infer/codeqwen_result_bruce_2.json'))

        valid_count = 0
        match_count = 0
        sql_error_count = 0
        df_match_count = 0
        print("------------Evaluating-------------")
        last_db_name = ''
        db_conn = None
        for idx, item in tqdm(enumerate(self.result_data)):
            gt_sql = item["sql"]
            pred_sql = extract_sql_from_qwen(item["pred_sql"])
            pred_sql = item["pred_sql"]
            item['db_name'] = "sql_eval_" + item['db_name'].split("sql_eval_")[-1]
            if self.sql_compare(item["sql"], item["pred_sql"], 'PostgreSQL') and False:# if we can directly match
                item["execution_match"] = 1
                item['df_match'] = 1
                match_count += 1
                valid_count += 1
                df_match_count += 1
            else:# we need execute it
                db_name = item['db_name']
                # db_conn = PG(db_name=item['db_name'])

                if not db_name==last_db_name:# we build another connection

                    if db_conn:# if we have the previous connection, we close it
                        db_conn.close()
                    # build new one
                    if self.sql_dialect == 'PostgreSQL':
                        db_conn = PG(db_name=item['db_name'])
                    elif self.sql_dialect == 'MySQL':
                        db_conn = MySQL(db_name=item['db_name'])
                    else:
                        raise NotImplementedError
                    last_db_name = db_name

                gt_res, err_gt = db_conn.fetch(gt_sql)
                pred_res, err_pred = db_conn.fetch(pred_sql)
                item['error'] = ''
                if not contains_none(gt_res) :
                    valid_count += 1
                    if pred_res is not None:
                        if result_eq(gt_res, pred_res, False):
                            match_count += 1
                            item["execution_match"] = 1
                        else:
                            item["execution_match"] = 0

                        if len(pred_res) > 0:
                            pred_col_num = len(pred_res[0])
                            gt_col_num = len(gt_res[0])
                            pred_df = pd.DataFrame([list(x) for x in pred_res],
                                        columns=['col_{}'.format(i) for i in range(1, pred_col_num+1)])
                            gt_df = pd.DataFrame([list(x) for x in gt_res],
                                         columns=['col_{}'.format(i) for i in range(1, gt_col_num + 1)])
                            if subset_df(gt_df, pred_df, query_category=''):
                                df_match_count += 1
                                item["df_match"] = 1
                            else:
                                item['df_match'] = 0
                        else:
                            item['df_match'] = 0

                    else:
                        sql_error_count+=1
                        item["execution_match"] = -1
                        item['df_match'] = -1
                        item['error'] = err_pred

                else:
                    item["execution_match"] = -2
                    item['df_match'] = -2

            self.result_data[idx] = item

        write_json(self.save_path + '/eval_result.json', self.result_data)
        print("--------------Evaluation Result---------------")
        print("total test num: ", len(self.result_data))
        print("valid test num: ", valid_count)
        print("sql error num: ", sql_error_count)
        print("accurate num: ", match_count)
        print("execution_match score: %.2f%%" % (100 * match_count/valid_count))
        print("executable score: %.2f%%" % (100 * (valid_count-sql_error_count) / valid_count))
        print("df match score: %.2f%%" % (100 * df_match_count/valid_count))
        print('evaluation results have been saved to path {}'.format(self.save_path))
        
if __name__ == "__main__":

    model_path = 'output/dense/0708_zc/train_bird_shucang_dpo_chase_ds40_pg2nl3k_bruce/checkpoint-7400/'
    prompt_type = 'PostgreSQL'
    infer_data_path = 'data/0628/sql_eval_chat_bruce.json'
    save_path = 'data/lzs_infer'

    evaluator = Evaluator(model_path, prompt_type, infer_data_path, save_path, 'auto')
    # evaluator = Evaluator('model/Qwen/CodeQwen1___5-7B-Chat', 'PostgreSQL', 'data/0628/sql_eval_chat_bruce.json', 'auto')
    # evaluator = Evaluator('output/dense/0624/qwen2_7b_5w_filter_shucang_dpo_group/checkpoint-13500', 'PostgreSQL', 'data/0621/sql_eval_0621.json', 'auto')
    # evaluator.inference()
    evaluator.evaluate()

