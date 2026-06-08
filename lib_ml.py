import numpy as np

# ฟังก์ชันคำนวณ Gini Impurity
def gini_impurity(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances) # weigth
    return gini

# ฟังก์ชันแบ่งข้อมูล
def split_data(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# select best features
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset)) #class --> [0,1,2]
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1): #row index
        for row in dataset:
            groups = split_data(index, row[index], dataset)
            gini = gini_impurity(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

# ฟังก์ชันสร้าง Terminal Node
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# ฟังก์ชันแบ่งโหนด (split node)
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

# Build Decision Tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

# ฟังก์ชันทำนายด้วย Decision Tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def training(dataset):
    # ตัวอย่างข้อมูล (Iris Dataset แบบย่อ)
    dataset = [
        [2.771244718, 1.784783929, 0],
        [1.728571309, 1.169761413, 0],
        [3.678319846, 2.81281357, 0],
        [3.961043357, 2.61995032, 0],
        [2.999208922, 2.209014212, 0],
        [7.497545867, 3.162953546, 1],
        [9.00220326, 3.339047188, 1],
        [7.444542326, 0.476683375, 1],
        [10.12493903, 3.234550982, 1],
        [6.642287351, 3.319983761, 1]
    ]

    # สร้างต้นไม้การตัดสินใจ
    tree = build_tree(dataset, max_depth=3, min_size=1)

    # ทดสอบการทำนาย
    for row in dataset:
        prediction = predict(tree, row)
        print(f"Expected: {row[-1]}, Predicted: {prediction}")