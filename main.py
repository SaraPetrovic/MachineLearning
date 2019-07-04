import pandas as pd
import numpy as np
import csv
import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score


def select_data(path):

    file_list = []
    with open(path, newline='', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        line = []
        for row in reader:
            if row[1] == 'trending_date':
                continue

            # video_id, trending_date, category_id, publish_time, views, likes, dislikes, comment_count, ratings_disabled, video_error_or_removed
            line = [row[0], row[1], row[3], row[4], row[5], row[7], row[8], row[9], row[10], row[13], row[14]]
            file_list.append(line)

    #print(file_list)

    # key -> song_id, value = [row_value, reps_num]
    dict = {}
    video_id = 0
    row_count = 0
    reps_num = 0
    new_date = ""
    row_data = ""
    r = 1
    for i in file_list:
        save_row = i
        video_id = i[0]
        row_count += 1
        new_date = i[1]
        count = 1
        r = 1
        if video_id in dict:
            continue
        for j in range(row_count, len(file_list)):
            row_data = file_list[j]
            if video_id == row_data[0]:
                if new_date == "":
                    continue
                date_obj = datetime.datetime.strptime(new_date, "%y.%d.%m")
                date_obj1 = date_obj + datetime.timedelta(days=1)
                row_date_obj = datetime.datetime.strptime(row_data[1], "%y.%d.%m")

                if date_obj1 == row_date_obj:
                    count = count + 1
                    new_date = row_data[1]
                    r = count
                else:
                    if reps_num < count:
                        reps_num = count
                        count = 1

        if r > reps_num:
            dict[video_id] = save_row
            dict[video_id].append(r)
        else:
            dict[video_id] = save_row
            dict[video_id].append(reps_num)

    #for v in dict.keys():
    #    print(dict[v])
    return dict

def save_to_file(path, list):
    with open(path, mode='w', encoding="utf8") as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['video_id', 'trending_date', 'category_id', 'publish_time', 'views', 'likes',
                            'dislikes', 'comment_count', 'ratings_disabled', 'video_error_or_removed', 'num_of_reps'])
        for i in list:
            date = i[4]
            if date == "":
                continue
            values = date.split("T")
            time = values[1][0:2]
            #print(time)
            writer.writerow([i[0], i[1], i[2], i[3], time, i[5], i[6], i[7], i[8], i[9], i[10], i[11]])

def read_data(path):
    data = pd.read_csv(path)

    data['ratings_disabled'].replace([False, True], [0, 1], inplace=True)
    data['video_error_or_removed'].replace([False, True], [0, 1], inplace=True)
    data['publish_time'].replace([00, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                                 inplace=True)

    y = data['num_of_reps']
    data = data.drop(labels='video_id', axis=1)
    data = data.drop(labels='trending_date', axis=1)
    x = data.drop(labels='num_of_reps', axis=1)
    return x, y

def read_data_model2(path):
    data = pd.read_csv(path)

    data['ratings_disabled'].replace([False, True], [0, 1], inplace=True)
    data['video_error_or_removed'].replace([False, True], [0, 1], inplace=True)
    data['publish_time'].replace(
        [00, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], inplace=True)

    y = data['views']
    data = data.drop(labels='video_id', axis=1)
    data = data.drop(labels='trending_date', axis=1)
    data = data.drop(labels='num_of_reps', axis=1)
    x = data.drop(labels='views', axis=1)
    return x, y

if __name__ == "__main__":
    values = select_data('CAvideos1.csv')
    values_test = select_data('CAvideos_test.csv')
    #print("values")

    save_to_file('new_file_CA.csv', list(values.values()))
    save_to_file('new_file_CA_test.csv', list(values_test.values()))
    #print("save to file")

    x_train, y_train = read_data('new_file_CA.csv')
    x_test, y_test = read_data('new_file_CA_test.csv')
    #print(y_test)

    clf1 = DecisionTreeClassifier(criterion='entropy', max_depth=1)
    clf2 = KNeighborsClassifier(n_neighbors=3)
    clf3 = svm.SVC(gamma='scale')

    tree_para = {'criterion': ['gini', 'entropy'], 'max_depth': [4, 5, 6, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90,
                                                                 120, 150]}
    gd_sr = GridSearchCV(clf1, param_grid=tree_para, cv=3, iid=True)

    # svm
    parameters = {'kernel': ['rbf'], 'C': [100], 'gamma': [0.01]}

    svc = svm.SVC(gamma="scale")
    classificator = GridSearchCV(clf3, parameters, cv=3, iid=True)

    # knn
    param_grid = {'n_neighbors': np.arange(1, 25)}
    knn_gscv = GridSearchCV(clf2, param_grid, cv=3, iid=True)

    bagging1 = BaggingClassifier(base_estimator=gd_sr, max_samples=0.7)
    bagging2 = BaggingClassifier(base_estimator=classificator, max_samples=0.7)
    bagging3 = BaggingClassifier(base_estimator=knn_gscv, max_samples=0.7)

    bagging1.fit(x_train, y_train)
    bagging2.fit(x_train, y_train)
    bagging3.fit(x_train, y_train)

    y_pred_b1 = bagging1.predict(x_test)
    y_pred_b2 = bagging2.predict(x_test)
    y_pred_b3 = bagging3.predict(x_test)

    values = [f1_score(y_test, y_pred_b1, average='micro'), f1_score(y_test, y_pred_b2, average='micro'),
              f1_score(y_test, y_pred_b3, average='micro')]

    print(values)