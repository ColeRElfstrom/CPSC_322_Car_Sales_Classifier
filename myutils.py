from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

def parallel_sort(distances, indices):
    indices_result = [x for _, x in sorted(zip(distances, indices))]
    distances_result = sorted(distances)
    return distances_result, indices_result
    
def compute_euclidean_distance(v1, v2):
    return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))

def dist_for_categorical(v1, v2):
    if v1 == v2:
        return 0
    else:
        return 1

def tdidt(current_instances, available_attributes, attribute_domains, class_index, header, case_3_instances = None, case_3_available_attributes = None):
    if case_3_instances == None:
        case_3_instances = current_instances
    if case_3_available_attributes == None:
        case_3_available_attributes = available_attributes.copy()
    # basic approach (uses recursion!!):
    #print("available attributes: ", available_attributes)
    # select an attribute to split on
    
    # ADDED FOR PROJECT BASED OFF NOTES IN ENSEMBLE FUN
    subset = compute_random_subset(available_attributes, len(available_attributes))
    split_attribute = select_attribute(current_instances, subset, class_index, header)
    # ADDED FOR PROJECT BASED OFF NOTES IN ENSEMBLE FUN


    #split_attribute = select_attribute(current_instances, available_attributes, class_index, header)
    #print("splitting on: ", split_attribute)
    available_attributes.remove(split_attribute)
    # cannot split on this attribute again in this branch of the tree
    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    att_index = header.index(split_attribute)
    partitions = partition_instances(current_instances, att_index, attribute_domains)
    #print("partitions:", partitions) # partitions is a dictionary

    
    
    # for each partition, repeat unless one of the following occurs (base case)
    #    CASE 1: all class labels of the partition are the same => make a leaf node
    #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
    #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
    total_length = 0
    for att_value, att_partition in partitions.items():
        total_length += len(att_partition)
    case_3_total_length = len(case_3_instances)
    for att_value in sorted(list(partitions.keys())):
        att_partition = partitions[att_value]
        value_subtree = ["Value", att_value]
        if len(att_partition) > 0 and same_class_label(att_partition, class_index):
            #print("Case 1, all same class label")
            # make a leaf node
            stats = compute_partition_stats(att_partition, class_index)[0]
            value_subtree.append(["Leaf", stats[0], stats[1], total_length])
            
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            #print("Case 2, clash")
            # make a leaf node selecting the majority vote value
            clash_stats = compute_partition_stats(att_partition, class_index)
            clash_frequency = []
            for clash_stat in clash_stats:
                clash_frequency.append(float(clash_stat[1]/clash_stat[2]))
            selected_clash = clash_frequency.index(max(clash_frequency))
            value_subtree.append(["Leaf", clash_stats[selected_clash][0], clash_stats[selected_clash][1], total_length])

        elif len(att_partition) == 0:
            case_3_split_attribute = select_attribute(case_3_instances, case_3_available_attributes, class_index, header)
            case_3_att_index = header.index(case_3_split_attribute)
            case_3_partitions = partition_instances(case_3_instances, case_3_att_index, attribute_domains)
            # backtrack and replace the attribute node with the majority vote value
            # overwrite tree to be a majority vote leaf node rather than an attribute node (tree = leaf node)
            for att_value in sorted(list(case_3_partitions.keys())):
                case_3_att_partition = case_3_partitions[att_value]
                clash_stats = compute_partition_stats(case_3_att_partition, class_index)
                clash_frequency = []
                for clash_stat in clash_stats:
                    clash_frequency.append(float(clash_stat[1]/clash_stat[2]))
                selected_clash = clash_frequency.index(max(clash_frequency))
                value_subtree = None
                tree = ["Leaf", clash_stats[selected_clash][0], total_length, case_3_total_length]
                break
        # none of the base cases were true... recurse
        else:
            #print("recursing")
            subtree = tdidt(att_partition, available_attributes.copy(), attribute_domains, class_index, header, current_instances.copy(), case_3_available_attributes.copy())
            value_subtree.append(subtree)
        if value_subtree != None:
            tree.append(value_subtree)
    return tree

def select_attribute(instances, attributes, class_index, header):
    entropys = []
    count = 0
    weighted_entropys = []
    for attribute in attributes: # att0 ....
        entropys.append([])
        domain = []
        for instance in instances:
            if instance[header.index(attribute)] not in domain:
                domain.append(instance[header.index(attribute)])
        for item in domain:
            group = []
            for instance in instances:
                if instance[header.index(attribute)] == item:
                    group.append(instance)
            stats = compute_partition_stats(group, class_index)
            entropy = []
            for stat in stats:
                entropy.append(-(stat[1]/stat[2])*np.log2((stat[1]/stat[2])))
            entropys[count].append(sum(entropy)) 
        count += 1

    for i, entropy_set in enumerate(entropys):
        count = 0
        stats = compute_partition_stats(instances, header.index(attributes[i]))
        entropy = []
        for stat in stats:
            entropy.append((stat[1]/stat[2])*(entropy_set[count]))
            count += 1
        weighted_entropys.append(sum(entropy))
    index = weighted_entropys.index(min(weighted_entropys))
    return attributes[index]

def partition_instances(instances, att_index, attribute_domains):
    # this is a group by attribute domain
    att_domain = attribute_domains["att" + str(att_index)]
    #print("attribute domain: ", att_domain)
    # lets use dictionaries
    partitions = {}
    for att_value in att_domain:
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)
    return partitions

def tdidt_predict(tree, instance, header):
    # are we at a leaf node (base case)
    # or an attribute node (need to recurse)
    info_type = tree[0] # Attribute or Leaf
    if info_type == "Leaf":
        # base case
        return tree[1] #label
    # if we are here we are at an attribute node
    # we need to figure out where in instance, this attribute's value is
    att_index = header.index(tree[1])
    # now loop through the value lists, looking for a match to instance[att_index]
    for i in range(2, len(tree)):
        value_list = tree[i]
        if value_list[1] == instance[att_index]:
            # we have a match, recurse on this values subtree
            return tdidt_predict(value_list[2], instance, header)
    
def same_class_label(instances, class_index):
    classes = []
    for instance in instances:
        if instance[class_index] not in classes:
            classes.append(instance[class_index])
    if len(classes) == 1:
        return True
    else:
        return False

def compute_partition_stats(instances, class_index):
    stats = []
    labels = []
    for instance in instances:
        if instance[class_index] not in labels:
            labels.append(instance[class_index])
    for label in labels:
        stats.append([label, 0, len(instances)])
    for instance in instances:
        for label in stats:
            if instance[class_index] == label[0]:
                label[1] += 1   
    return stats

def print_tree(tree, header, attribute_names, class_name, string):
    info_type = tree[0]
    if info_type == "Leaf":
        # base case
        print(string + " THEN " + class_name + " == " + str(tree[1]))
        return

    att_index = header.index(tree[1])
    for i in range(2, len(tree)):
        value_list = tree[i]
        if string == "IF ":
            print_tree(value_list[2], header, attribute_names, class_name, string + str(attribute_names[att_index])+ " == " + str(value_list[1]))
        else: 
            print_tree(value_list[2], header, attribute_names, class_name, string + " AND " + str(attribute_names[att_index]) + " == " + str(value_list[1]))


def discretize_sales_price(prices):
    """Divides the given sales price into a range of 1-10
    Args:
        value
    Returns:
        discretization (int)
    """
    discritized_prices = []
    max_price = max(prices)
    min_price = min(prices)
    avg_price = mean(prices)
    print(avg_price)
    print(min_price)
    print(max_price)
    for price in prices:
        discritized_prices.append(individual_discretize(price))

    ranges = [0, 2500, 5000, 7500, 12500, 20000, 35000, 50000, 75000, 100000, 200000]

    return discritized_prices, ranges

def individual_discretize(price):
    if price >= 200000:
        return 10
    if price >= 100000:
        return 9
    if price >= 75000:
        return 8
    if price >= 50000:
        return 7
    if price >= 35000:
        return 6
    if price >= 20000:
        return 5
    if price >= 12500:
        return 4
    if price >= 7500:
        return 3
    if price >= 5000:
        return 2
    if price >= 2500:
        return 1
    else:
        return 0

def clean_data(table):
    sales = table.get_column("pricesold")
    remove_indexes = []
    for index, sale in enumerate(sales):
        if sale == 0:
            remove_indexes.append(index)
    table.drop_rows(remove_indexes)

    return table

def plot_prices(prices, ranges):
    class_counts = []
    for value in range(11):
        class_counts.append(int(prices.count(value)))
    print(class_counts)
    xrng = np.arange(len(ranges))
    print(xrng)
    plt.figure()
    plt.bar(xrng, class_counts)
    plt.xticks(xrng, ranges, rotation=45)

def compute_random_subset(values, num_values):
    values_copy = values[:] # shallow copy
    np.random.shuffle(values_copy) # in place shuffle
    return values_copy[:num_values]
    
def compute_bootstrapped_sample(table):
    n = len(table)
    # np.random.randint(low, high) returns random integers from low (inclusive) to high (exclusive)
    sampled_indexes = [np.random.randint(0, n) for _ in range(n)]
    sample = [table[index] for index in sampled_indexes]
    out_of_bag_indexes = [index for index in list(range(n)) if index not in sampled_indexes]
    out_of_bag_sample = [table[index] for index in out_of_bag_indexes]
    return sample, out_of_bag_sample