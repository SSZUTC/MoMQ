import sys
sys.path.append('../')
from database.Nebula import NebulaGraphClient
from database.Neo4j import Neo4jClient
from transformers.trainer_pt_utils import LabelSmoother
import transformers
from transformers import (
    GenerationConfig,
    set_seed
)
from trainer.trainer import MyTrainer
from trainer.argument import ModelArguments, TrainingArguments, DataArguments, LoraArguments
from trainer.train_util import load_tokenizer_and_model, DataCollatorForGeneration
from datasets import load_dataset, concatenate_datasets
import numpy as np
import sqlglot
from sqlglot.optimizer import optimize
from sqlglot import parse_one, diff
from sqlglot.diff import Keep
from database.PG import PG
from database.Mysql import MySQL
from utils.value_match import result_eq, df_sim_pair, get_jw_distance
from utils.df_eval import subset_df
import pandas as pd
from utils.common_utils import read_json, write_json
from datasets import set_caching_enabled
import json

set_caching_enabled(True)
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
call_counter = 0  # 全局变量

def sql_compare(sql_gt, sql_pred, dialect):
        
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

def contains_none(res: list[tuple]) -> bool:
    if res is None:
        return True
    if len(res) == 0:
        return True
    else:
        for res_tuple in res:# for each tuple
            if len(res_tuple) > 0 and None in res_tuple:
                return True
    return False

def calculate_sql_metrics(label_sqls, pred_sqls, eval_data, eval_result_map):
    # print("------------Evaluating-------------")
    last_db_name = ''
    db_conn = None
    for idx in range(len(label_sqls)):
        # print(idx, len(label_sqls), len(eval_data))
        split_info = label_sqls[idx].split('[sep]')
        db_name = split_info[0]
        sql_type = split_info[1]
        
        if sql_type not in ['postgresql', 'mysql']:
            continue

        if sql_type == 'postgresql':
            db_name = "sql_eval_" + db_name.split("sql_eval_")[-1]

        gt_sql = split_info[-1]
        pred_sql = pred_sqls[idx]
        item = eval_data[idx]
        item['pred_sql'] = pred_sql

        # we build another connection
        if not db_name==last_db_name:
            # if we have the previous connection, we close it
            if db_conn:
                db_conn.close()
            # build new one
            if sql_type == 'postgresql':
                db_conn = PG(db_name=db_name)
            elif sql_type == 'mysql':
                db_conn = MySQL(db_name=db_name)
            else:
                raise NotImplementedError
            last_db_name = db_name

        gt_res, err1 = db_conn.fetch(gt_sql)
        pred_res, err2 = db_conn.fetch(pred_sql)
        item['error'] = ''
        if not contains_none(gt_res):
            eval_result_map[sql_type]['valid_count'] += 1
            eval_result_map[sql_type]['similarity'] += get_jw_distance(gt_sql, pred_sql)
            if pred_res is not None:
                if result_eq(gt_res, pred_res, False):
                    eval_result_map[sql_type]['match_count'] += 1
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
                        eval_result_map[sql_type]['df_match_count'] += 1
                        item["df_match"] = 1
                    else:
                        item['df_match'] = 0
            else:
                eval_result_map[sql_type]['sql_error_count'] += 1
                item["execution_match"] = -1
                item['df_match'] = -1
                item['error'] = err2
        else:
            item["execution_match"] = -2
            item['df_match'] = -2
        eval_data[idx] = item

def calculate_nosql_metrics(label_sqls, pred_sqls, eval_data, eval_result_map):
    neo4j_client = {
        'fincen': Neo4jClient(uri='bolt://127.0.0.1:7687'),
        'movies': Neo4jClient(uri='bolt://127.0.0.1:8687')
    }
    nebula = NebulaGraphClient()
    for idx in range(len(label_sqls)):
        split_info = label_sqls[idx].split('[sep]')
        db_name = split_info[0]
        sql_type = split_info[1]
        
        if sql_type not in ['cypher', 'ngql']:
            continue
        
        item = eval_data[idx]
        
        gt_sql = split_info[-1]
        pred_sql = pred_sqls[idx]

        item['pred_sql'] = pred_sql

        if sql_type == 'cypher':
            gt_result, err_gt = neo4j_client[db_name].fetch(gt_sql)
            pred_result, err_pred = neo4j_client[db_name].fetch(pred_sql)
        elif sql_type == 'ngql':           
            gt_result, err_gt = nebula.fetch(gt_sql, db_name)
            pred_result, err_pred = nebula.fetch(pred_sql, db_name)

        eval_result_map[sql_type]['valid_count'] += 1
        eval_result_map[sql_type]['similarity'] += get_jw_distance(gt_sql, pred_sql)

        if pred_result is not None and err_pred is None:
            try:
                if df_sim_pair((gt_sql, gt_result), (pred_sql, pred_result)) == 1:
                    eval_result_map[sql_type]['match_count'] += 1
                    item["execution_match"] = 1
                else:
                    item["execution_match"] = 0
            except:
                print(item)
                print((gt_sql, gt_result))
                print((pred_sql, pred_result))
                item["execution_match"] = 0

        else:
            eval_result_map[sql_type]['sql_error_count'] += 1
            item["execution_match"] = -1
            item['df_match'] = -1
            item['error'] = err_pred
        eval_data[idx] = item

def sample_data(data, sample_map):
    sample_map = json.loads(sample_map)
    sample_data = None
    for sql_type in sample_map:
        records = data.filter(lambda example: example['sql_type'] == sql_type)
        data_num = min(sample_map[sql_type], len(records))
        records = records.select(range(data_num))
        if sample_data is None:
            sample_data = records
        else:
            sample_data = concatenate_datasets([sample_data, records])

    # total_records = len(sample_data)
    # remainder = total_records % 8
    
    # if remainder != 0:
    #     sample_data = sample_data.select(list(range(total_records - remainder)))
    return sample_data

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    # set seed
    set_seed(training_args.seed)
    # load tokenizer and model
    tokenizer, model = load_tokenizer_and_model(model_args, training_args, lora_args)

    train_data_raw = load_dataset('json', data_files=data_args.data_path)['train']
    eval_data_raw = load_dataset('json', data_files=data_args.eval_data_path)['train']
    # sample data
    if data_args.train_dialects_num_map:
        train_data_raw = sample_data(train_data_raw, data_args.train_dialects_num_map)
    if data_args.eval_dialects_num_map:
        eval_data_raw = sample_data(eval_data_raw, data_args.eval_dialects_num_map)
    print('train_data_raw', train_data_raw)
    print('eval_data_raw', eval_data_raw)

    log_first = True
    def preprocess_data(example):
        conversations = example['conversations']
        sql_type = example.get("sql_type", "postgresql")
        text = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            # chat_template=TEMPLATE,
            add_generation_prompt=False
        )
        nonlocal log_first
        if log_first:
            # print(text)
            log_first = False

        prompt_part = tokenizer.apply_chat_template(
            conversations[:1],
            # chat_template=TEMPLATE,
            tokenize=False,
            add_generation_prompt=True
        )
        encodings = tokenizer(text, truncation=True)
        target_idx = len(tokenizer(prompt_part)['input_ids'])
        # target = conversations[1]['content']
        # idx = text.find(target)
        # target_idx = encodings.char_to_token(idx)
        labels = encodings['input_ids'].copy()
        
        if target_idx:
            labels[:target_idx] = [IGNORE_TOKEN_ID]*target_idx
        else:
            print(len(encodings['input_ids']))
            print(text)
            print(f'train example len >= {training_args.model_max_length} !!!!')

        assert len(labels) == len(encodings['input_ids'])
        if training_args.enable_dialect_router:
            if sql_type.lower() == 'postgresql':
                labels[0] = 0
            elif sql_type.lower() == 'mysql':
                labels[0] = 1
            elif sql_type.lower() == 'cypher':
                labels[0] = 2
            elif sql_type.lower() == 'ngql':
                labels[0] = 3
        encodings['labels'] = labels
        return encodings
    
    def preprocess_eval_data(example):
        conversations = example['conversations'][:1]
        text = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            # chat_template=TEMPLATE,
            add_generation_prompt=True
        )
        encodings = tokenizer(text, truncation=True)
        db_name = example['db_name']
        # sql_type = example['sql_type']
        sql_type = example.get("sql_type", "postgresql")
        target = example['conversations'][1]['content']
        labels = tokenizer(db_name + '[sep]' + sql_type + '[sep]' + target, truncation=True)['input_ids']
        encodings['labels'] = labels
        return encodings


    train_data = train_data_raw.map(preprocess_data, remove_columns=train_data_raw.column_names, num_proc=16, load_from_cache_file=True)
    eval_data = eval_data_raw.map(preprocess_eval_data, remove_columns=eval_data_raw.column_names, num_proc=16, load_from_cache_file=True)
    data_collator = DataCollatorForGeneration(tokenizer)
    # data_collator = DataCollatorForGeneration(tokenizer, padding='max_length', max_length=training_args.model_max_length)
    
    generation_config = GenerationConfig(
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id,
        max_new_tokens = 512,
        temperature = None,
        num_beams = 1,
        top_p = None,
        do_sample = False,
        use_cache = True
    )

    def calcute_sql_ex(pred_sqls, label_sqls):
        global call_counter
        call_counter += 1
        eval_data_result = [row for row in eval_data_raw]
        eval_result_map = {
            'postgresql': {},
            'mysql': {},
            'cypher': {},
            'ngql': {}
        }
        
        for key in eval_result_map:
            eval_result_map[key]['valid_count'] = 0
            eval_result_map[key]['match_count'] = 0
            eval_result_map[key]['df_match_count'] = 0
            eval_result_map[key]['sql_error_count'] = 0
            eval_result_map[key]['similarity'] = 0

        calculate_sql_metrics(label_sqls, pred_sqls, eval_data_result, eval_result_map)
        calculate_nosql_metrics(label_sqls, pred_sqls, eval_data_result, eval_result_map)


        metrics = {}
        save_path = training_args.output_dir + '/' + model_args.model_name_or_path.replace('/', '_')
        for key in eval_result_map:
            if eval_result_map[key]['valid_count'] == 0:
                continue
            ex = 100 * round(eval_result_map[key]['match_count'] / eval_result_map[key]['valid_count'], 5)
            df_ex = 100 * round(eval_result_map[key]['df_match_count'] / eval_result_map[key]['valid_count'], 5)
            executable = 100 * round((eval_result_map[key]['valid_count'] - eval_result_map[key]['sql_error_count']) / eval_result_map[key]['valid_count'], 5)
            similarity = 100 * round(eval_result_map[key]['similarity'] / eval_result_map[key]['valid_count'], 5)

            save_path += f'_{ex}'
            metrics[f'eval_{key}_ex'] = ex
            metrics[f'eval_{key}_df_ex'] = df_ex
            metrics[f'eval_{key}_executable'] = executable
            metrics[f'eval_{key}_similarity'] = similarity

        save_path += "_" + str(int(call_counter * training_args.eval_steps))
        save_path += '.json'
        write_json(save_path, eval_data_result)
        return metrics

    def compute_metrics(eval_pred):
        predictions, labels, inputs = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        gt_sqls = tokenizer.batch_decode(labels, skip_special_tokens=True)
        pred_sqls = [
            pred[len(prompt):] for prompt, pred in zip(decoded_inputs, decoded_preds)
        ]
        metrics = calcute_sql_ex(pred_sqls, gt_sqls)
        return metrics
    
    training_args.generation_config = generation_config
    # Add extra loss
    extra_losses = []
    if training_args.enable_dialect_router:
        extra_losses.append('dialect_loss')
    if training_args.output_router_logits:
        extra_losses.append('aux_loss')
    # Start trainner
    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        extra_losses=extra_losses
    )
    trainer.train(resume_from_checkpoint=training_args.resume)
    # metrics = trainer.evaluate()
    # trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)
    trainer.save_state()
    trainer.save_model()
if __name__ == "__main__":
    train()
