import os
import jieba
import random

"""
函数说明:中文文本处理
 
Parameters:
    folder_path - 文本存放的路径
    test_size - 测试集占比，默认占所有数据集的百分之20
Returns:
    all_words_list - 按词频降序排序的训练集列表
    train_data_list - 训练集列表
    test_data_list - 测试集列表
    train_class_list - 训练集标签列表
    test_class_list - 测试集标签列表
"""
def TextProcessing(folder_path, test_size = 0.2):
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []
    
    # 存取每个txt数据集文件
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        files = os.listdir(new_folder_path)

        j = 1

        # 遍历每个txt数据集文件
        for file in files:
            if j > 100:
                break
            with open(os.path.join(new_folder_path, file), 'r', encoding = 'utf-') as f:
                raw = f.read()
            word_cut = jieba.cut(raw, cut_all = False)
            word_list = list(word_cut)

            data_list.append(word_list)
            class_list.append(folder)
            j += 1

        data_class_list = list(zip(data_list, class_list))
        random.shuffle(data_class_list)
        index = int(len(data_class_list) * test_size) + 1
        train_list = data_class_list[index:]
        test_list = data_class_list[:index]
        train_data_list, train_class_list  =zip(*train_list)
        test_data_list, test_class_list = zip(*test_list)

        all_words_dict = {}
        
        print(data_list)
        print(class_list)

if __name__ == '__main__':
    folder_path = 'SogouC/Sample'
    TextProcessing(folder_path)
