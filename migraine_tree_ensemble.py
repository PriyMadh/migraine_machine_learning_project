import pandas as pd
import numpy as np

# load data
df = pd.read_csv('my_migraine_history.csv')

#  Define order of feature columns
feature_names = [
    'hard_workout', 
    'processed_food', 
    'water_ounces', 
    'sun_minutes', 
    'hours_of_sleep',
    'screen_time_hours'
]

# get X matrix and y vector
X = df[feature_names].to_numpy()
y = df['target_migraine'].to_numpy()

#  Verify everything loaded correctly
print("--- DATA ---")
print(f"Total tracking rows (days) loaded: {X.shape[0]}")
print(f"Total feature columns loaded: {X.shape[1]}")
print(f"Target column shape: {y.shape}")

print("\nVerify Row 1 (Day 1) contents:")
for name, val in zip(feature_names, X[0]):
    print(f"  {name}: {val}")
print(f"  Target (Migraine?): {y[0]}")

def compute_entropy(y):
    if len(y) ==0:
        return 0
    
    else:
        p_1 = np.mean(y)
        if p_1 == 1 or p_1 == 0: # if the split was perfect entropy is minimum
            entropy = 0
        else:
            p_0 = 1.0 - p_1
            entropy = - (p_1 * np.log2(p_1) + p_0 * np.log2(p_0)) #entropy formula

    return entropy

def split_data(X, feature_idx, threshold):
    feature_vals = X[:, feature_idx] # grab column of feature values
    # take the values of the features and see if they are above or below a threshold
    left_mask = feature_vals <= threshold # is this value for a feature below a threshold
    right_mask = ~left_mask # opposite boolians of left_mask

    return left_mask, right_mask

def compute_info_gain(y, left_mask, right_mask):
    starting_entropy = compute_entropy(y)
    n_total = len(y)
    n_left = np.sum(left_mask) #counts the True values in left mask
    n_right = np.sum(right_mask) # counts true in right mask

    if n_left ==0 or n_right==0:
        return 0.0
    
    # basically checking what the values of y are on either when you split on this feature based on a thrshold and grouping them into the left and right node
    y_left = y[left_mask] # when these values for the feature were below a threshold or above a threshold (true/false) it checks what y was (what actually hapened to y on the days that this condition was met) # drops the ones that aren't true ( say the array was 1, 0, 0, 1 and the left mask got everything true as somehting that was 1, then that would get t, f, f, t, and this would return 1, 1; anf for the right side it would get true for everythig that was 0 and that would return f, t, t, f, and then 0,0)
    y_right = y[right_mask] # so here its keeping the values of y when it was above a threshold

    # find entropy for both sides
    entropy_left = compute_entropy(y_left)
    entropy_right = compute_entropy(y_right)

    post_split_entropy = (n_left / n_total) * entropy_left + (n_right / n_total) * entropy_right

    info_gain = starting_entropy - post_split_entropy # diff in entropy

    return info_gain

# find split wiht the most info gain
def find_best_spit(X, y):
    best_gain = -1
    best_feature_idx = None
    best_threshold  = None

    n_features = X.shape[1]

    for feature_idx in range (n_features): # run this for each column of features
        feature_vals = X[:, feature_idx] # grab that whole column
        unique_vals = np.sort(np.unique(feature_vals)) # don't want to deal with repeats 

        midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2.0 # everything but last one added to everything but first one then avergaeged out to find the midpoints between all of them

        for threshold in midpoints: # find the best threshold to split on (one wiht most infor gain)
            # compare info gains for every feature, and wiithin those feature rounds, every threshold (important bc most of the features here aren't binary)
            left_mask, right_mask = split_data(X, feature_idx, threshold)
            gain  = compute_info_gain(y, left_mask, right_mask)

            if gain > best_gain:
                best_gain = gain
                best_feature_idx = feature_idx
                best_threshold = threshold

    return best_feature_idx, best_threshold

# set up for random forest algo to increase generalization/reduce overfitting
def sampling_with_replacement(X, y):
    samples = X.shape[0]
    random_idxs = np.random.choice(samples, size = samples, replace = True) # sample randomly from the data set with replcament

    return X[random_idxs], y[random_idxs] # X matrix with some recurring examples adn y vector with some recurring examples

def build_tree(X, y, current_depth = 0, max_depth = 3): # depth sort of arbitrary here but I feel like 3-4 is a good option
    
    samples, features = X.shape
    
    if len(np.unique(y)) == 1 or current_depth >= max_depth or samples < 2: # if it is full of only one type of y label, reaches max depth, or has only one example in the node: stop
        return int(np.round(np.mean(y))) # return 1 or 0 
    
    #find best feature and threshold used for split
    best_feature_idx, best_threshold = find_best_spit(X, y)

    # if no split increases info gain
    if best_feature_idx is None:
        return int(np.round(np.mean(y)))
    
    #split the data based on if it is above a threshold or below for a certain feature based on the calculated best feature and best threshold
    left_mask, right_mask = split_data(X, best_feature_idx, best_threshold)

    # recursively call the tree function to continually split based on the best feature and with the best threshold (unless max depth reached, not enough samples in the node, or the node is all one label)
    left_branch = build_tree(X[left_mask], y[left_mask], current_depth + 1, max_depth)
    right_branch = build_tree(X[right_mask], y[right_mask], current_depth + 1, max_depth)

    #dictionary of all the values for each split
    return {
        'current_depth': current_depth,
        'feature_idx': best_feature_idx,
        'threshold': best_threshold,
        'left': left_branch,   
        'right': right_branch  
    }

# build tree ensemble of 10 trees with random forest algo
my_trees = []

for tree_idx in range(10):

    X_sample, y_sample = sampling_with_replacement(X, y) # sample with replacment
    tree = build_tree(X_sample, y_sample, current_depth = 0, max_depth = 3) # build the tree unitil parameters reached
    my_trees.append(tree) # add tree to ensemble
    print(f"current_depth, best_feature_idx, Best_threshold, left_branch, and right_branch for tree {tree}: {my_trees[tree_idx]}")






