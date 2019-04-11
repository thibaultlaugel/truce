from collections import Counter
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances_argmin_min, pairwise_distances

from exploration import growingspheres as gs
from lash import Hill
from lore import get_adversarial

import load_dataset



####### MACROS

def generate_inside_ball_new(center, segment, n):
    def norm(v):
        return np.linalg.norm(v, ord=2, axis=1)
    d = center.shape[0]
    z = np.random.normal(0, 1, (n, d))
    u = np.random.uniform(segment[0]**d, segment[1]**d, n)
    r = u**(1/float(d))
    z = np.array([a * b / c for a, b, c in zip(z, r,  norm(z))])
    z = z + center
    return z


def find_majority(votes):
    vote_count = Counter(votes)
    top_two = vote_count.most_common(2)
    if len(top_two)>1 and top_two[0][1] == top_two[1][1]:
        return 0
    return top_two[0][0]


def local_detection_test(obs_to_interprete):
    X_train_ennemies_sorted = sorted(zip(X_train_ennemies, pairwise_distances(obs_to_interprete.reshape(1,-1), X_train_ennemies)[0]), key=lambda x: x[1])
    CF = X_train_ennemies_sorted[0][0]

    N_BALL = 10000

    i, j, k = -1, 0, 0
    while True:
        a = X_train_ennemies_sorted[i][1]
        b = X_train_ennemies_sorted[j][1]
        if k == 0:
            a = 0
        else:
            a = a - epsilon
        ball = generate_inside_ball_new(obs_to_interprete, (a, b), n=N_BALL)
        ball_pred = clf.predict(ball)
        layer = ball[np.where(ball_pred == clf.predict(CF.reshape(1,-1)))]
        if k == 0:
            ball_ennemies = layer
            print(ball_pred.mean())
            if (ball_pred.mean() == 1.0) or (ball_pred.mean() == 0.0):
                print('cric crac')
                N_BALL = 20000
                ball = generate_inside_ball_new(obs_to_interprete, (a, b), n=N_BALL)
                ball_pred = clf.predict(ball)
                if (ball_pred.mean() == 1.0) or (ball_pred.mean() == 0.0):
                    print('dans ma baraque')
                    return -2, 1
                layer = ball[np.where(ball_pred == clf.predict(CF.reshape(1,-1)))]
        else:
            ball_ennemies = np.append(previous_ennemies, layer, axis=0)
            
        if k == 0:
            kgraph = kneighbors_graph(ball, n_neighbors=1, mode='distance', metric='euclidean', include_self=False, n_jobs=-1)
            closest_distances = kgraph.toarray()[np.where(kgraph.toarray()>0)]
            epsilon = closest_distances.max()
        ball_ennemies = np.insert(ball_ennemies, 0, X_train_ennemies_sorted[j][0], axis=0)
        clustering = DBSCAN(eps=epsilon, min_samples=2, leaf_size=30, n_jobs=-1).fit(ball_ennemies)
        labels = clustering.labels_
        if k == 0:
            candidates_pool = ball_ennemies
            n_candidates = candidates_pool.shape[0]

        unique_labels = list(set(labels))
        labels_connected = list(set(labels[:j+1]))
        for p in range(len(labels_connected)):
            if labels_connected[p] == -1:
                labels_connected[p] = max(unique_labels) + 1
                unique_labels.append(max(unique_labels) + 1)

        ennemies_connected = ball_ennemies[np.where(np.isin(labels, labels_connected))]

        candidates_pool = np.array([x for x in candidates_pool if x not in ennemies_connected])

        idx_candidate_pools = np.where(np.isin(ball_ennemies, candidates_pool).prod(axis=1))
        labels_OG_candidates_pool = list(set(labels[idx_candidate_pools]))
        new_shape = ball_ennemies[np.where(np.isin(labels, labels_OG_candidates_pool))].shape
        
        if k > 0:
            if candidates_pool.shape[0] ==  0: 
                n_ucf = candidates_pool.shape[0]
                n_jcf = n_candidates - n_ucf
                return n_jcf, n_ucf

            if (new_shape == prev_shape) or (k == X_test.shape[0] - 1): 
                n_ucf = candidates_pool.shape[0]
                n_jcf = n_candidates - n_ucf
                return n_jcf, n_ucf
        prev_shape = new_shape


        previous_ennemies = ball_ennemies
        i += 1
        j += 1
        k += 1
        


def evaluation_test(adv):
    idx_ally_adv, delta = pairwise_distances_argmin_min(adv.reshape(1,-1), X_train_ennemies, metric='euclidean')
    CF_adv = X_train_ennemies[idx_ally_adv][0]
    delta_adv = pairwise_distances(adv.reshape(1, -1), X_train_ennemies).min()
    N_BALL = 15000
    RADIUS = delta_adv + 0.0
    ball = generate_inside_ball_new(adv, (0, RADIUS), n=N_BALL)
    ball_pred = clf.predict(ball)
    
    ball_allies_adv = ball[np.where(ball_pred == clf.predict(adv.reshape(1,-1)))]
    ball_allies_adv = np.insert(ball_allies_adv, 0, CF_adv, axis=0)
    ball_allies_adv = np.insert(ball_allies_adv, 0, adv, axis=0)
    kg = kneighbors_graph(ball_allies_adv, n_neighbors=1, mode='distance', metric='euclidean', include_self=False, n_jobs=-1)
    closest_distances = kg.toarray()[np.where(kg.toarray()>0)]
    eps = closest_distances.max()
    clustering = DBSCAN(eps=eps, min_samples=2, leaf_size=30, n_jobs=-1).fit(ball_allies_adv)
    labels = clustering.labels_  
    justif = int(labels[0] == labels[1])
    return justif

        
        
##################################
'''
DATASETS
'''
##################################

### MOONS
#X, y = datasets.make_moons(n_samples = 1000, shuffle=True, noise=0.2, random_state=3)


### WINE
'''wine = datasets.load_wine()
X = wine.data
y = wine.target
y = y[np.where(y < 2)]
X = X[np.where(y < 2)]
'''

### BOSTON
'''from sklearn import datasets
X, y = datasets.load_boston(return_X_y=True)
X = X - X.mean(axis=0)
X = X / list(map(max, zip(abs(X.max(axis=0)), abs(X.min(axis=0)))))
y = np.array([int(x) for x in y>26.0])
vars_ = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'BLACK', 'LSTAT']
vars_ = list(enumerate(vars_))
'''

### RECIDIVISM
'''df = pd.read_csv('/home/laugel/Documents/thesis/code/highgarden/highgarden/datasets/recidivism.csv', sep=',', header=0)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#df = df[df['score_text'] != 'Low']
df = df[(df['days_b_screening_arrest'] >= -30) & (df['days_b_screening_arrest']<=30)]
df = df[df['is_recid'] != -1]
#y = df.score_text.astype('category').cat.codes
y = (df.score_text != 'Low').astype('int')
del df['score_text']
del df['decile_score']
del df['violent_recid']
del df['r_days_from_arrest']
del df['id']
del df['decile_score.1']
del df['v_decile_score']
del df['start']
del df['end']
X = df.select_dtypes(include=numerics)
X['race'] = df['race']
X['sex'] = df['sex']
X = pd.get_dummies(X)
del X['race_Caucasian']
del X['sex_Male']
vars_ = X.columns
print(X.head())
X = np.array(X)
'''

### GERMAN CREDIT
#X, y = load_dataset.main('credit', n_obs=10000)


### ONLINE NEWS POPULARITY
X, y = load_dataset.main('news', n_obs=10000)


#normalisation rajoutee elle n y etait pas attention
X = (X.copy() - X.mean(axis=0))/X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
#clf = xgb.XGBClassifier().fit(X_train, y_train)
clf = RandomForestClassifier(200, random_state=0).fit(X_train, y_train)
#clf = GaussianNB().fit(X_train, y_train)
#clf = SVC(C=1.0, probability=True).fit(X_train, y_train)
#clf = KNeighborsClassifier(n_neighbors=15, metric='manhattan').fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(clf)
print('accuracy ', sum(y_pred == y_test)/len(y_test))
print('auc ', roc_auc_score(y_test, y_pred))







#################################
'''
MAIN LOOP
'''
#################################


R_MIN = 0.25

R_list, s_list = [], []
Jlash_list, JGS05_list, Jlore_list = [], [], []
Dlash_list, DGS05_list, Dlore_list = [], [], []
n_risk = 0

for i in range(0, min(50, X_test.shape[0])):
    print('==', i, '==')
    obs_to_interprete = X_test[i]
    X_train_ennemies = X_train[np.where((y_train != clf.predict(obs_to_interprete.reshape(1,-1))[0]) & (clf.predict(X_train) == y_train) )]
    delta = pairwise_distances(obs_to_interprete.reshape(1, -1), X_train_ennemies).min()
    class_obs = clf.predict(obs_to_interprete.reshape(1,-1))
    target_class = int(1 - class_obs)
    
    R_k, s_k = [], []
    J_lash_k, J_gs05_k, J_lore_k = [], [], []
    D_lash_k, D_gs05_k, D_lore_k = [], [], []
    for k in range(5):
        n_jcf, n_ucf = local_detection_test(obs_to_interprete)
        #print(n_jcf, n_ucf)
        R = n_ucf / (n_jcf + n_ucf)
        s = int(n_ucf > 0)
        #print(R)
        R_k.append(R)
        s_k.append(s)
        
        adv_lash = Hill(obs_to_interprete, B=delta*1.0)
        DGS = gs.RobustGrowingSpheres1(obs_to_interprete, prediction_fn=clf.predict_proba, target_proba=0.5)
        adv_gs05 = DGS.find_counterfactual()
	adv_lore = lore.get_adversarial(obs_to_interprete, prediction_fn=clf.predict_proba)
        advs = [adv_lash, adv_gs05, adv_lore]
        
        if R >= R_MIN:
            if k == 0:
                n_risk += 1
            advs_justif = []
            advs_dist = []
            for adv in advs:
                justif_score = evaluation_test(adv)
                print('justif score', justif_score)
                advs_justif.append(justif_score)
                distance = float(pairwise_distances(obs_to_interprete.reshape(1,-1), adv.reshape(1,-1))[0])
                advs_dist.append(distance)
        else:
            advs_justif = [-99] * 2
            advs_dist = [-99] * 2
        justif_lash, justif_gs05, justif_lore = advs_justif
        dist_lash, dist_gs05, dist_lore = advs_dist
        J_lash_k.append(justif_lash)
        J_gs05_k.append(justif_gs05)
        D_lash_k.append(dist_lash)
        D_gs05_k.append(dist_gs05)
	J_lore_k.append(justif_lore)
	D_lore_k.append(dist_lore)
    
    R_k = np.array(R_k)
    R_list.append(R_k[np.where(R_k >= 0)].mean())
    s_list.append(find_majority(s_k))
    J_lash_k = np.array(J_lash_k)
    J_gs05_k = np.array(J_gs05_k)
    J_lore_k = np.array(J_lore_k)
    Jlash_list.append(find_majority(J_lash_k))
    JGS05_list.append(find_majority(J_gs05_k))
    Jlore_list.append(find_majority(J_lore_k))
    
    D_gs05_k = np.array(D_gs05_k)
    D_lash_k = np.array(D_lash_k)
    D_lore_k = np.array(D_lore_k)
    Dlash_list.append(D_lash_k[np.where(D_lash_k >= 0)].mean())
    DGS05_list.append(D_gs05_k[np.where(D_gs05_k >= 0)].mean())
    Dlore_list.append(D_lore_k[np.where(D_lore_k >= 0)].mean())

R_list = np.array(R_list)
s_list = np.array(s_list)
Jlash_list = np.array(Jlash_list)
Dlash_list = np.array(Dlash_list)
JGS05_list = np.array(JGS05_list)
DGS05_list = np.array(DGS05_list)
Jlore_list = np.array(Jlore_list)
Dlore_list = np.array(Dlore_list)

print('Average risk score :', R_list[np.where(R_list >= 0.0)].mean(), '(', R_list[np.where(R_list >= 0.0)].std(), ')')
print('Proportion of vulnerable examples:', s_list.mean())
print('Number with R > 0.25 :', n_risk)





####################################
'''
OUTPUT
'''
####################################


FOLDER_EXPE_RESULTS = './results/'

fname = 'detection_evaluation'
dataset = 'news'
model = 'RF200'
f_out = FOLDER_EXPE_RESULTS + '_'.join((fname, dataset, model)) + '.txt'

model_accuracy = sum(y_pred == y_test)/len(y_test)

mean_R_score = R_list[np.where(R_list >= 0.0)].mean()
std_R_score = R_list[np.where(R_list >= 0.0)].std()
mean_s_score = s_list.mean()
n_R_risk = n_risk
mean_J_lash = Jlash_list[np.where(Jlash_list >= 0.0)].mean()
std_J_lash = Jlash_list[np.where(Jlash_list >= 0.0)].std()
mean_J_GS05 = JGS05_list[np.where(JGS05_list >= 0.0)].mean()
std_J_GS05 = JGS05_list[np.where(JGS05_list >= 0.0)].std()
mean_J_lore = Jlore_list[np.where(Jlore_list >= 0.0)].mean()
std_J_lore = Jlore_list[np.where(Jlore_list >= 0.0)].std()
mean_D_lash = Dlash_list[np.where(Dlash_list >= 0.0)].mean()
std_D_lash = Dlash_list[np.where(Dlash_list >= 0.0)].std()
mean_D_GS05 = DGS05_list[np.where(DGS05_list >= 0.0)].mean()
std_D_GS05 = DGS05_list[np.where(DGS05_list >= 0.0)].std()
mean_D_lore = Dlore_list[np.where(Dlore_list >= 0.0)].mean()
std_D_lore = Dlore_list[np.where(Dlore_list >= 0.0)].std()


out = {
    'dataset': dataset,
    'classifier': model,
    'model_accuracy': model_accuracy,
    'mean_R_score': mean_R_score,
    'std_R_score': std_R_score,
    'mean_S_score': mean_s_score,
    'n_R_risk': n_R_risk,
    'mean_J_lash': mean_J_lash,
    'std_J_lash': std_J_lash,
    'mean_J_GS05': mean_J_GS05,
    'std_J_GS05': std_J_GS05,
    'mean_J_lore': mean_J_lore,
    'std_J_lore': std_J_lore,
    'mean_D_lash': mean_D_lash,
    'std_D_lash': std_D_lash,
    'mean_D_GS05': mean_D_GS05,
    'std_D_GS05': std_D_GS05,
    'mean_D_lore': mean_D_lore,
    'std_D_lore': std_D_lore,

}

with open(f_out, 'w') as f:
    line = str(out)
    f.write(line)


print("===============")
print(R_list)
print(Jlash_list)
