import sys
import random

import numpy as np
import pandas as pd
import copy

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


###############################################################################
############################### Label Flipping ################################
###############################################################################

def attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n):
    # TODO: You need to implement this function!
    # You may want to use copy.deepcopy() if you will modify data
    number_to_flip = int(n*len(y_test))
    indicies_to_flip=random.sample(range(0, len(y_test)), number_to_flip)
    y_train_= copy.deepcopy(y_train)
    y_train_[indicies_to_flip] = (y_train_[indicies_to_flip] +1)%2
    acur = 0
    for i in range(100):
        if model_type == "DT":
            myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
            myDEC.fit(X_train, y_train_)
            DEC_predict = myDEC.predict(X_test)
            acur += accuracy_score(y_test, DEC_predict)

        if model_type == "LR":
            myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
            myLR.fit(X_train, y_train_)
            LR_predict = myLR.predict(X_test)
            acur +=(accuracy_score(y_test, LR_predict))

        if model_type == "SVC":
            mySVC = SVC(C=0.5, kernel='poly', random_state=0,probability=True)
            mySVC.fit(X_train, y_train_)
            SVC_predict = mySVC.predict(X_test)
            acur+= accuracy_score(y_test, SVC_predict)





    return acur/100

###############################################################################
############################## Inference ########################################
###############################################################################

def inference_attack(trained_model, samples, t):
    # TODO: You need to implement this function!
    prob=trained_model.predict_proba(samples)
    max_conf =np.max(prob,axis=1)
    conf = max_conf>t
    TP = len(samples[conf])
    FN = len(samples) - len(samples[conf])


    return TP/(TP+FN)

###############################################################################
################################## Backdoor ###################################
###############################################################################

def backdoor_attack(X_train, y_train, model_type, num_samples):    
    # TODO: You need to implement this function!
    # You may want to use copy.deepcopy() if you will modify data
    indicies_normally_zero = y_train ==0
    indicies_backdoored = np.random.choice(np.array(np.nonzero(indicies_normally_zero*1))[0], num_samples)
    to_be_backdoored=X_train[indicies_backdoored]

    indicies_backdoored_tested = np.random.choice(np.array(np.nonzero(indicies_normally_zero*1))[0], 5)
    to_be_backdoored_=X_train[indicies_backdoored_tested]
    X_test = copy.deepcopy(to_be_backdoored_)

    X_test[0][len(X_train[1])-1]=4
    X_test[1][len(X_train[1])-1]=4
    X_test[2][len(X_train[1])-1]=4
    X_test[3][len(X_train[1])-1]=4
    X_test[4][len(X_train[1])-1]=4
    if num_samples == 0:
        return 0
    back_dor= copy.deepcopy(to_be_backdoored)
    X_train_ = copy.deepcopy(X_train)
    y_train_ = copy.deepcopy(y_train)
    back_dor[:,len(back_dor[0])-1] = 4
    y_back = np.zeros(len(back_dor))+1
    Synth_Dataset = np.concatenate((X_train_,back_dor),axis=0)

    Synth_y = np.concatenate((y_train_,y_back),axis=0)


    if model_type == "DT":
        myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
        myDEC.fit(Synth_Dataset, Synth_y)
        DEC_predict = myDEC.predict(X_test)
        return np.sum(DEC_predict,axis=0)/5

    if model_type == "LR":
        myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
        myLR.fit(Synth_Dataset, Synth_y)
        LR_predict = myLR.predict(X_test)
        return np.sum(LR_predict,axis=0)/5

    if model_type == "SVC":
        mySVC = SVC(C=0.5, kernel='poly', random_state=0,probability=True)
        mySVC.fit(Synth_Dataset, Synth_y)
        SVC_predict = mySVC.predict(X_test)
        return np.sum(SVC_predict,axis=0)/5



    return -999



###############################################################################
############################## Evasion ########################################
###############################################################################





def evade_model(trained_model, actual_example):
    # TODO: You need to implement this function!
    actual_class = trained_model.predict([actual_example])[0]
    pred_class= actual_class
    modified_example = copy.deepcopy(actual_example)
    while pred_class == actual_class:
        if actual_class ==1:
            for i in range(len(actual_example)):

                if i == 2:
                    modified_example[i] -= 0.02
                if i == 3:
                    modified_example[i] += 0.02
                if i == 4:
                    modified_example[i] -= 0.1


                """
                if i == 3:
                    modified_example[i] += 0.1
                """


                """
                if i == 5:
                    modified_example[i] -=0.1
                if i == 6:
                    modified_example[i] -=0.1
                if i == 7:
                    modified_example[i] -=0.1
                if i == 8:
                    modified_example[i] -=0.1
                
                if i == 9:
                    modified_example[i] -=0.1
                """
        if actual_class ==0:
            for i in range(len(actual_example)):
                """
                if i == 0:
                    modified_example[i] += 0.1
                if i == 1:
                    modified_example[i] -=0.1
                """

                """
                if i == 3:
                    modified_example[i] -=0.1
                """


                if i == 2:
                    modified_example[i] += 0.02
                if i == 3:
                    modified_example[i] -= 0.02
                if i == 4:
                    modified_example[i] += 0.1


                """
                if i == 5:
                    modified_example[i] +=0.1
                if i == 6:
                    modified_example[i] +=0.1
                if i == 7:
                    modified_example[i] +=0.1
                if i == 8:
                    modified_example[i]+=0.1
            
                if i == 9:
                    modified_example[i] += 0.1
                """
        pred_class = trained_model.predict([modified_example])

        # do something to modify the instance
    #    print("do something")
    return modified_example

def calc_perturbation(actual_example, adversarial_example):
    # You do not need to modify this function.
    if len(actual_example) != len(adversarial_example):
        print("Number of features is different, cannot calculate perturbation amount.")
        return -999
    else:
        tot = 0.0
        for i in range(len(actual_example)):
            tot = tot + abs(actual_example[i]-adversarial_example[i])
        return tot/len(actual_example)

###############################################################################
############################## Transferability ################################
###############################################################################

def evaluate_transferability(DTmodel, LRmodel, SVCmodel, actual_examples):
    # TODO: You need to implement this function!
    print("Here, you need to conduct some experiments related to transferability and print their results...")
    trained_models = [DTmodel, LRmodel, SVCmodel]
    model_types = ["DT", "LR", "SVC"]
    for a,trained_model in enumerate(trained_models):
        modified_examples =np.zeros(actual_examples.shape)
        for indx,exampls in enumerate(actual_examples):
            adversarial_example = evade_model(trained_model, exampls)
            modified_examples[indx]= copy.deepcopy(adversarial_example)
        for j,trained_model_compare in enumerate(trained_models):
            if a!=j:
                trans_counter= 0
                for indx2,exampls2 in enumerate(modified_examples):
                    if trained_model.predict([modified_examples[indx2]]) == trained_model_compare.predict([modified_examples[indx2]]):
                        trans_counter+=1
                print("The transferibility from ",model_types[a]," to ", model_types[j], " is ", 100*(trans_counter/40))





###############################################################################
########################## Model Stealing #####################################
###############################################################################

def steal_model(remote_model, model_type, examples):
    # TODO: You need to implement this function!
    # This function should return the STOLEN model, but currently it returns the remote model
    # You should change the return value once you have implemented your model stealing attack

    labels=remote_model.predict(examples)

    if model_type == "DT":
        myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
        myDEC.fit(examples, labels)
        return myDEC
    if model_type == "LR":
        myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
        myLR.fit(examples, labels)
        return myLR

    if model_type == "SVC":
        mySVC = SVC(C=0.5, kernel='poly', random_state=0,probability=True)
        mySVC.fit(examples, labels)
        return  mySVC

    return remote_model
    

###############################################################################
############################### Main ##########################################
###############################################################################

## DO NOT MODIFY CODE BELOW THIS LINE. FEATURES, TRAIN/TEST SPLIT SIZES, ETC. SHOULD STAY THIS WAY. ## 
## JUST COMMENT OR UNCOMMENT PARTS YOU NEED. ##

def main():
    data_filename = "forest_fires.csv"
    features = ["Temperature","RH","Ws","Rain","FFMC","DMC","DC","ISI","BUI","FWI"]
    
    df = pd.read_csv(data_filename)
    df = df.dropna(axis=0, how='any')
    df["DC"] = df["DC"].astype('float64')
    y = df["class"].values
    y = LabelEncoder().fit_transform(y)    
    X = df[features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0) 
    

    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
    myDEC.fit(X_train, y_train)
    DEC_predict = myDEC.predict(X_test)
    print('Accuracy of decision tree: ' + str(accuracy_score(y_test, DEC_predict)))

    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
    myLR.fit(X_train, y_train)
    LR_predict = myLR.predict(X_test)
    print('Accuracy of logistic regression: ' + str(accuracy_score(y_test, LR_predict)))
    
    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0,probability=True)
    mySVC.fit(X_train, y_train)
    SVC_predict = mySVC.predict(X_test)
    print('Accuracy of SVC: ' + str(accuracy_score(y_test, SVC_predict)))

    model_types = ["DT", "LR", "SVC"]

    # Label flipping attack executions:
    model_types = ["DT", "LR", "SVC"]
    n_vals = [0.05, 0.10, 0.20, 0.40]
    for model_type in model_types:
        for n in n_vals:
            acc = attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n)
            print("Accuracy of poisoned", model_type, str(n), ":", acc)

    # Inference attacks:
    samples = X_train[0:100]
    t_values = [0.99,0.98,0.96,0.8,0.7,0.5]
    for t in t_values:
        print("Recall of inference attack", str(t), ":", inference_attack(mySVC,samples,t))
    
    # Backdoor attack executions:
    counts = [ 0,1, 3, 5, 10]
    for model_type in model_types:
        for num_samples in counts:
            success_rate = backdoor_attack(X_train, y_train, model_type, num_samples)
            print("Success rate of backdoor:", success_rate, "model_type:", model_type, "num_samples:", num_samples)
    
    #Evasion attack executions:
    trained_models = [myDEC, myLR, mySVC]
    model_types = ["DT", "LR", "SVC"] 
    num_examples = 40
    for a,trained_model in enumerate(trained_models):
        total_perturb = 0.0
        for i in range(num_examples):
            actual_example = X_test[i]
            adversarial_example = evade_model(trained_model, actual_example)
            if trained_model.predict([actual_example])[0] == trained_model.predict([adversarial_example])[0]:
                print("Evasion attack not successful! Check function: evade_model.")
            perturbation_amount = calc_perturbation(actual_example, adversarial_example)
            total_perturb = total_perturb + perturbation_amount
        print("Avg perturbation for evasion attack using", model_types[a] , ":" , total_perturb/num_examples)

    
    # Transferability of evasion attacks:
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 40
    evaluate_transferability(myDEC, myLR, mySVC, X_test[0:num_examples])
    
    # Model stealing:
    budgets = [8, 12, 16, 20, 24]
    for n in budgets:
        print("******************************")
        print("Number of queries used in model stealing attack:", n)
        stolen_DT = steal_model(myDEC, "DT", X_test[0:n])
        stolen_predict = stolen_DT.predict(X_test)
        print('Accuracy of stolen DT: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_LR = steal_model(myLR, "LR", X_test[0:n])
        stolen_predict = stolen_LR.predict(X_test)
        print('Accuracy of stolen LR: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_SVC = steal_model(mySVC, "SVC", X_test[0:n])
        stolen_predict = stolen_SVC.predict(X_test)
        print('Accuracy of stolen SVC: ' + str(accuracy_score(y_test, stolen_predict)))
    

if __name__ == "__main__":
    main()
