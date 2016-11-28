#encoding=utf-8
raw_data = [
    [[0,0], [0]],
    [[0,10], [1]],
    [[10,0], [1]],
    [[10,10], [0]]
]
def normal(raw_data):
    normal_data=[]
    max_no=-9999
    min_no=9999
    for ele in raw_data:#获取最大最小值
        ele_list=ele[0]
        des=ele[1][0]
        if des > max_no:
            max_no=des
        if des < min_no:
            min_no=des
        for ele2 in ele_list:
            if ele2 > max_no:
                max_no=ele2
            if ele2 < min_no:
                min_no= ele2
    # print(max_no)
    # print(min_no)
    for ele in raw_data:#开始转换
        ele_list = ele[0]
        des = ele[1][0]
        des_normal= 2*(des-min_no)/(max_no-min_no) - 1#目标值进行归一化
        des_list=[]
        des_list.append(des_normal)
        y_list = []
        for x in ele_list:
            y=2*(x-min_no)/(max_no-min_no) - 1
            y_list.append(y)
        tmp_list=[]
        tmp_list.append(y_list)
        tmp_list.append(des_list)
        normal_data.append(tmp_list)
    for ele in normal_data:#输出结果
        print(ele)

normal(raw_data)