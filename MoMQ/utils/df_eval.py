import pandas as pd
from pandas._testing import assert_series_equal, assert_frame_equal

def normalize_table(
        df: pd.DataFrame, query_category: str, sql: str = None
) -> pd.DataFrame:
    """
    将结果集dataframe归一化
    1. 去除掉重复的行
    2. 将列按字母顺序排序
    3. 对行按值排序 if query_category is not 'order_by'
    4. 重置索引index
    """
    # remove duplicate rows, if any
    df = df.drop_duplicates()

    # 先根据列名对列排序
    sorted_df = df.reindex(sorted(df.columns), axis=1)

    # 测试用例中该sql最终结果集是否要order_by
    has_order_by = False
    if query_category == "order_by":
        has_order_by = True

    # 如果不需要order_by的，
    if not has_order_by:
        # sort rows using values from first column to last
        sorted_df = sorted_df.sort_values(by=list(sorted_df.columns))

    # reset index
    sorted_df = sorted_df.reset_index(drop=True)
    return sorted_df


def subset_df(
        df_sub: pd.DataFrame,
        df_super: pd.DataFrame,
        query_category: str
) -> bool:
    """
    判定df_sub是否是df_super的子集
    @param df_sub: gt的sql的结果集
    @param df_super: 生成的sql的结果集
    @param query_category: order_by 用于标识结果集dataframe是否要排序
    @param query_sub: gt
    @param query_super: gen_qsl
    @return:
    """
    if df_sub.empty:
        return False  # handle cases for empty dataframes

    df_super_temp = df_super.copy(deep=True)
    matched_columns = []
    for col_sub_name in df_sub.columns:
        col_match = False
        for col_super_name in df_super_temp.columns:
            col_sub = df_sub[col_sub_name].sort_values().reset_index(drop=True)
            col_super = (
                df_super_temp[col_super_name].sort_values().reset_index(drop=True)
            )
            try:
                assert_series_equal(
                    col_sub, col_super, check_dtype=False, check_names=False
                )
                col_match = True
                matched_columns.append(col_super_name)
                # remove col_super_name to prevent us from matching it again
                df_super_temp = df_super_temp.drop(columns=[col_super_name])
                break
            except AssertionError:
                continue
        if col_match == False:
            return False
    df_sub_normalized = normalize_table(df_sub, query_category)

    # get matched columns from df_super, and rename them with columns from df_sub, then normalize
    df_super_matched = df_super[matched_columns].rename(
        columns=dict(zip(matched_columns, df_sub.columns))
    )
    df_super_matched = normalize_table(
        df_super_matched, query_category
    )

    try:
        assert_frame_equal(df_sub_normalized, df_super_matched, check_dtype=False)
        return True
    except AssertionError:
        return False

#
# def result_df_compare(gt_res, pred_res):
#     assert len(gt_res) == len(pred_res)
#     for
#

if __name__ == "__main__":
    # case 1:生成的sql排序字段选择错误(gmv_amt vs gmt_amt1)导致顺序不对了 此时虽然target列(shop_city_name)的值一样都是上海北京杭州但应当判定错误
    q1 = "23年三、四月期间运单gmv最高的TOP3城市是？"
    gt_sql1 = """select 
    shop_city_name
from t_ads
where ds between '20230301' and '20230430'
group by shop_city_name
order by sum(gmv_amt)
desc limit 3;"""
    gen_sql1 = """select 
    shop_city_name,sum(gmv_amt1) as gmv_amt
from t_ads
where ds between '20230301' and '20230430'
group by shop_city_name
order by gmv_amt
desc limit 3;"""
    gt_df1 = pd.DataFrame([["上海"], ["北京"], ["杭州"]], columns=["gtdf_col1"])
    gen_df1 = pd.DataFrame([["北京", 6], ["上海", 5], ["杭州", 4]], columns=["gendf_col1", "gendf_col2"])
    # 结果集是要排序的
    query_category = 'order_by'
    # 归一化后的 gt_df1 = [["上海"], ["北京"], ["杭州"]] ; gen_df1 = [["北京"], ["上海"], ["杭州"]] 结果为false
    res1 = subset_df(gt_df1, gen_df1, query_category, gt_sql1, gen_sql1)
    assert res1 is False

    # case 2:生成的sql字段多了一列但是主列正确，此时判定正确
    q2 = "23年三、四月期间运单gmv大于10000的的城市是？"
    gt_sql2 = """select 
        shop_city_name
    from t_ads
    where ds between '20230301' and '20230430'
    group by shop_city_name
    having sum(gmv_amt) > 10000
    ;"""
    gen_sql2 = """select 
        shop_city_name,sum(gmv_amt) as gmv_amt
    from t_ads
    where ds between '20230301' and '20230430'
    group by shop_city_name
    having gmv_amt > 10000
    ;"""
    gt_df1 = pd.DataFrame([["上海"], ["北京"], ["杭州"]], columns=["gtdf_col1"])
    gen_df1 = pd.DataFrame([["杭州", 4], ["上海", 6], ["北京", 5]], columns=["gendf_col1", "gendf_col2"])
    # 结果集不需要排序，值对即可
    query_category = ''
    # 归一化后的 gt_df1 = [["上海"], ["北京"], ["杭州"]] ; gen_df1 = [["上海"], ["北京"], ["杭州"]] 结果为True
    res2 = subset_df(gt_df1, gen_df1, query_category, gt_sql1, gen_sql1)
    assert res2 is True