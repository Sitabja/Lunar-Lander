from IPython.display import display, HTML, Image

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
from sklearn import tree
from sklearn import metrics
from sklearn import tree
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import linear_model

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import itertools
from pandas.plotting import scatter_matrix
from collections import OrderedDict




# Create a new classifier which is based on the sckit-learn BaseEstimator and ClassifierMixin classes
class SuperLearnerClassifier(BaseEstimator, ClassifierMixin):
    
    """An ensemble classifier that uses heterogeneous models at the base layer and a aggregatnio model at the aggregation layer. A k-fold cross validation is used to gnerate training data for the stack layer model.

    Parameters
    ----------
     criteria: str,'label'or'probability', default:'label'
                  used to specify whether the stacked model input is a label prediction or probability predictions of base models
        
     stack_input: bool, default:False
                     used to specify whether to add original input to the stack layer.
        
     stack_model : str, name of the model at the stack model, default:'CART'
     base_models: list, list of base models, default:['LR','KNN','CART','SVC','NB','MLP']
                     used to specify the list of base models to be used.
                     
     number_of_models: int, number of base models, default:6
                          used to specify the number of base models
        
    Methods
    ----------
     fit(X,Y) : Fit the model according to the given training data.
     
     predict(X): Predict class labels for samples in X.
     
     predict_proba: Predict class probabilities for samples in X.
     
     get_predictive_power_base_models(): to get the accuracies of the base models on the training set of the Superlearner
     
     get_confusion_matrix_base_models(): to get the confusion matrices of the base learners on the training set of the Superlearner
     
     get_diversity_base_models() :  to get the diversity among the base models in terms of correlation coefficient, Q statistics 
                                 and disagreement measure
    Attributes:
    -------------
    base_models_predictions : the list of predictions of the abse models.
    
    
   Notes
    -----
    Number of base mdoels should be between 5 to 10.
    The list of base models should be from the choice 7 base models : ['LR','KNN','CART','SVC','NB','MLP','RF']
    If a list of base models are passed then Superlearners trains the base _models with default sensible hyper parameters.
    if a dictionary of base models along with the hyper parameters are sent, it overwrites the default hyper parameters of 
    the base models.
    
    
    See also
    --------
    ----------
    .. [1]  van der Laan, M., Polley, E. & Hubbard, A. (2007). 
            Super Learner. Statistical Applications in Genetics 
            and Molecular Biology, 6(1) 
            doi:10.2202/1544-6115.1309
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> clf = SuperLearnerClassifier()
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)

    """
    
    # Constructor for the classifier object
    def __init__(self,criteria='label',stack_input=False, stack_model= 'CART',
                 base_models=['LR','KNN','CART','SVC','NB','MLP'], number_of_models=6):
                                                                            
        
        """Setup a SuperLearner classifier .
        Parameters
        ----------
        criteria: str,'label'or'probability', default:'label'
                  used to specify whether the stacked model input is a label prediction or probability predictions of base models
        
        stack_input: bool, default:False
                     used to specify whether to add original input to the stack layer.
        
        base_models: list or dict, list of base models or dictionary of base models with their corresponding hyper parameters, default:['LR','KNN','CART','SVC','NB']
                     used to specify the list of base models to be used.
                     
        number_of_models: int, number of base models, default:5
                          used to specify the number of base models
        Returns
        -------
        self: Object
        """
        if criteria not in ['label','probability']:
            raise ValueError('Unknown criteria',criteria);
            return;
            
        if number_of_models < 5 or number_of_models > 10:
            raise ValueError('Number of models should be 5 to 10');
            return;
        
        if len(base_models) < 3:
            raise ValueError('Number of models should be 5 to 10');
            return;
        
        #default hyper parameter if no hyper parameter is used
        best_parameters = {'CART': {'criterion': 'entropy', 'max_depth': 6, 'min_samples_split': 50}, 
                           'RF': {'max_features': 4, 'min_samples_split': 25, 'n_estimators': 200}, 
                           'KNN': {'n_neighbors': 6}, 
                           'LR': {'C': 0.2, 'max_iter': 1000, 'multi_class': 'ovr', 'solver': 'liblinear'}, 
                           'MLP': {'alpha': 1e-05, 'hidden_layer_sizes': (400, 200), 'solver':'lbfgs', 'activation':'logistic'}, 
                           'SVC': {'probability':True},
                           'NB':{}
                          }
        
        
        self.base_models = base_models
        self.criteria = criteria
        self.stack_input = stack_input
        self.stack_model = stack_model
        self.base_models_predictions = []
          
        if(not isinstance(base_models, list)):
            # overwrite the default parameters with the parameters passed in the base_models dictionary:
            for model in self.base_models:
                best_parameters[model] = self.base_models[model];
            self.base_models = list(self.base_models.keys())
        
        self.models = []
        self.number_of_models = number_of_models;
        
        #make a dictionary of model choices
        models_choices_ = {}
        models_choices_['LR'] = LogisticRegression(**best_parameters['LR'])
        models_choices_['SVC'] = SVC(**best_parameters['SVC'])
        models_choices_['KNN'] =  KNeighborsClassifier(**best_parameters['KNN'])
        models_choices_['CART'] = DecisionTreeClassifier(**best_parameters['CART'])
        models_choices_['NB'] = GaussianNB() #no hyper parameter to tune for Naive Bayes
        models_choices_['RF'] = RandomForestClassifier(**best_parameters['RF'])
        models_choices_['MLP'] = MLPClassifier(**best_parameters['MLP'])
        
        #get the actuale base models to be used in the superlearner based on the number of models and model choices
        index = 0
        for n in range(number_of_models):
            if index == len(self.base_models):
                index = 0
            if self.base_models[index] not in models_choices_:
                raise ValueError('Bad choice of model ',self.base_models[index])
                return
            else:
                self.models.append(copy.deepcopy(models_choices_[self.base_models[index]]))
            index+=1;
            
            
        #stack layer model       
        if(self.stack_model not in models_choices_):
            raise ValueError('Bad choice of model ',self.stack_model)
            return
            
       
        self.final_stack_model = copy.deepcopy(models_choices_[self.stack_model])
       
        

    # The fit function to train a classifier
    def fit(self, X, y):
        """Build a SuperLearner classifier from the training set (X, y).
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples. 
        y : array-like, shape = [n_samples] 
            The target values (class labels) as integers or strings.
        Returns
        -------
        self : object
        """     
        k = 10
        each_fold_length = int(len(X)/k)
        
        # fold X and Y
        sub_samples_X=[X[i:i+each_fold_length] for i in range(0,len(X),each_fold_length)]
        sub_samples_y=[y[i:i+each_fold_length] for i in range(0,len(X),each_fold_length)]
   
        stacked_predictions = []
        for i in range(0,len(sub_samples_X)):
            # make the ith fold as test fold and remaining folds as training set
            X_test = sub_samples_X[i];
            X_train = [sub_samples_X[j] for j in range(0,len(sub_samples_X)) if j!=i]
            y_test = sub_samples_y[i];
            y_train = [sub_samples_y[j] for j in range(0,len(sub_samples_X)) if j!=i]
            
            #flatten the X_train and y_train to match the shape
            X_train = list(itertools.chain.from_iterable(X_train))
            y_train = list(itertools.chain.from_iterable(y_train))
            print("fold ",i)
            #fit the base models for the ith test_fold in k-fold
            y_pred = []
            for model in self.models:
                model.fit(X_train,y_train); print("done ", model)
				
                #depending upon the criterian taking prediction label/ prediction probability from the trained base modls
                if self.criteria == 'label':
                    predicted_value = model.predict(X_test)
                else:
                    predicted_value = model.predict_proba(X_test)
                y_pred.append(predicted_value.tolist())
            
            #stack the predicted output of the base models.
            stacked_predictions.append(np.column_stack(y_pred).tolist())
            
            
        # flatten the stacked_prediction to match the shape
        stacked_predictions = list(itertools.chain.from_iterable(stacked_predictions))
        
        #base_models_predictions is stored to verify diversity among base models
        self.base_models_predictions = np.column_stack(stacked_predictions)
       
        #depending upon the parameter the original input data is stacked
        if(self.stack_input):
            stacked_predictions = np.column_stack((X,stacked_predictions))
       
        #fit the stacked model 
        self.final_stack_model.fit(stacked_predictions,y)
    
        #train the base models again with the entire training set
        for model in self.models:
                model.fit(X,y)
        #storing the y values to verify the predictive power of the base models
        self.y = y
        
        return self

    # The predict function to make a set of predictions for a set of query instances
    def predict(self, X):
        """Predict class labels of the input samples X.
        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The input samples. 
        Returns
        -------
        p : array of shape = [n_samples, ].
            The predicted class labels of the input samples. 
        """
        X_predict = []
        for model in self.models:
            if self.criteria == 'probability':
                X_predict.append(model.predict_proba(X))
            else:
                X_predict.append(model.predict(X))
                
        stacked_predict = np.column_stack(X_predict).tolist()

        if(self.stack_input):
            stacked_predict = np.column_stack((X,stacked_predict))
            
       
        final_predict = self.final_stack_model.predict(stacked_predict)
        
        return final_predict
    
    # The predict function to make a set of predictions for a set of query instances
    def predict_proba(self, X):
        """Predict class probabilities of the input samples X.
        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The input samples. 
        Returns
        -------
        p : array of shape = [n_samples, n_labels].
            The predicted class label probabilities of the input samples. 
        """
        X_predict = []
        for model in self.models:
            if self.criteria == 'probability':
                X_predict.append(model.predict_proba(X).tolist())
            else:
                X_predict.append(model.predict(X).tolist())
        stacked_predict = np.column_stack(X_predict)
        if(self.stack_input):
            stacked_predict = np.column_stack((X,stacked_predict))
        final_predict = self.final_stack_model.predict_proba(stacked_predict)

        return final_predict
    
    """
    We want the each base model to be a strong predictor
    """
    def get_predictive_power_base_models(self):
        """ Measure the predictive power of the base_models
        Parameters
        ----------
        self : Object
        
        Returns
        -------
        predictive_power: A dictionary containing the accuracy score all the base models
        """
        predictive_power = {}
        for indx in range(len(self.base_models_predictions)):
            predictive_power[self.base_models[indx]] = metrics.accuracy_score(self.y, self.base_models_predictions[indx])
        return predictive_power
    
    def get_confusion_matrix_base_models(self):
        confusion_matrics_dict = {}
        for i in range(len(self.base_models)):
            confusion_matrics_dict[self.base_models[i]] = pd.crosstab(np.array(self.y), self.base_models_predictions[i], rownames=['True'], colnames=['Predicted'], margins=True, dropna = False)
        return confusion_matrics_dict
    
    def get_pairwise_relation_(self,model1_predict,model2_predict):
        """ Measure the number of agreements and disagreements between two models
        Parameters
        ----------
        model1_predict , model2_predict : Number of predictions of two models
        
        Returns
        -------
        A tuple containing (N00,N01,N10,N11)
        
        """
        N00 = 0
        N01 = 0;
        N10 = 0;
        N11 = 0
        for i in range(len(model1_predict)):
            if(model1_predict[i] == y_train[i] and model2_predict[i] == y_train[i]):
                N11 +=1
            elif(model1_predict[i] == y_train[i] and model2_predict[i] != y_train[i]):
                N10 += 1
            elif(model1_predict[i] != y_train[i] and model2_predict[i] == y_train[i]):
                N01 += 1
            elif(model1_predict[i] != y_train[i] and model2_predict[i] != y_train[i]):
                N00 += 1
        return (N00,N11,N10,N01)
    
    def get_q_score_(self,N00,N11,N10,N01):
        """ Measure the Q_Statistics between two models
        Parameters
        ----------  
        NOO: Number of samples incorrectly predicted by model1 as well as model2 
        N01: Number of samples incorrectly predicted by model1, but correctly predicted by model2
        N10: Number of samples correctly predicted by model1, but incorrectly predicted by model2
        N11: Number of sample correclt predicted by both the models
        
        Returns
        -------
        q_measure : Q_statistics measure
        
        """
        q_measure = (N00*N11 - N10*N01)/(N00*N11 + N10*N01)
        return q_measure
    
    def get_corr_coef_(self,N00,N11,N10,N01):
        """ Measure the Q_Statistics between two models
        Parameters
        ----------  
        NOO: Number of samples incorrectly predicted by model1 as well as model2 
        N01: Number of samples incorrectly predicted by model1, but correctly predicted by model2
        N10: Number of samples correctly predicted by model1, but incorrectly predicted by model2
        N11: Number of sample correclt predicted by both the models
        
        Returns
        -------
        corr_coeff : Correlation Coefficient
        
        """
        corr_coeff = (N00*N11 - N10*N01)/((N11+N10)*(N01+N00)*(N11+N01)*(N10+N00))**0.5
        return corr_coeff
    
    def get_disagreement_measure_(self,N00,N11,N10,N01):
        """ Measure the Q_Statistics between two models
        Parameters
        ----------  
        NOO: Number of samples incorrectly predicted by model1 as well as model2 
        N01: Number of samples incorrectly predicted by model1, but correctly predicted by model2
        N10: Number of samples correctly predicted by model1, but incorrectly predicted by model2
        N11: Number of sample correclt predicted by both the models
        
        Returns
        -------
        dm : Disagreement measure
        
        """
        dm = (N01+N10)/(N00+N11+N10+N01)
        return dm;
    
    """
        In any ensemble (eg.Superlearner) we want the classifiers to be weakly correlated to each other.
        There are three measures are taken to find out the diversities among the model.
        
        The Q Statistics :
            The Q statistics of two binary classifier outputs (correct/incorrect), yi and yk
            Qi,k = (N11*N00 − N01*N10)/(N11*N00 + N01*N10) Classifiers that tend to recognize the same objects correctly will have positive
            values of Q, and those which commit errors on different objects will render Q negative.
        
        The correlation coefficient ρ :
            The correlation between two binary classifier outputs (correct/incorrect), yi and yk , is
            ρi,k = (N11*N00 − N01*N10)/((N11 + N10)(N01 + N00)(N11 + N01)(N10 + N00))**0.5
        
        The disagreement measure:
            It is the ratio between the number of observations on which one classifiervis correct and the other is incorrect to the total number of observations.
            Disi,k = (N01 + N10) / (N11 + N10 + N01 + N00)
            
        NOO: Number of samples incorrectly predicted by model1 as well as model2 
        N01: Number of samples incorrectly predicted by model1, but correctly predicted by model2
        N10: Number of samples correctly predicted by model1, but incorrectly predicted by model2
        N11: Number of sample correclt predicted by both the models
        
        ref : https://link.springer.com/content/pdf/10.1023%2FA%3A1022859003006.pdf
        
        """
    def get_diversity_base_models(self):
        """ Measure the diversities among all the base models taking into acount their agreement and disagreement.
        Parameters
        ----------
        self: Object
        
        Returns
        -------
        R : A tuple of dataframe containing pairwise Q_statistics, pairwise Correlation Coefficient, pairwise disagreement measure
        
        """
        corr_coef_matrix = {}
        q_stat_matrix = {}
        dm_matrix = {}
        for i in range(len(self.base_models)):
            list_of_corr_coef = []
            list_of_q_stat = []
            list_of_dm = []
            for j in range(len(self.base_models)):
                    model1 = self.base_models[i]
                    model2 = self.base_models[j]
                    model1_predictions = self.base_models_predictions[i]
                    model2_predictions = self.base_models_predictions[j]
                
                    N00, N01, N10, N11 = self.get_pairwise_relation_(model1_predictions, model2_predictions)
                    
                    # get correlation coefficient 
                    list_of_corr_coef.append(self.get_corr_coef_(N00, N01, N10, N11))
                    #get q_statistics
                    list_of_q_stat.append(self.get_q_score_(N00, N01, N10, N11))
                    #get disagreement measure
                    list_of_dm.append(self.get_disagreement_measure_(N00, N01, N10, N11))
            
            corr_coef_matrix[self.base_models[i]] = list_of_corr_coef
            q_stat_matrix[self.base_models[i]] = list_of_q_stat
            dm_matrix[self.base_models[i]] = list_of_dm
            
        return (pd.DataFrame(OrderedDict(corr_coef_matrix),index=self.base_models), \
                pd.DataFrame(OrderedDict(q_stat_matrix),index=self.base_models), \
                pd.DataFrame(OrderedDict(dm_matrix),index=self.base_models))
    