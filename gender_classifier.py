from nltk.classify import NaiveBayesClassifier
from nltk.classify import accuracy
from sklearn.utils import shuffle


class GenderClassifier:

    def gender_features(self, name):
        """
        Traing and Testing feature sets
        """
        
        return ({
                'last_is_vowel': (name[-1] in 'AEIOUY'),
                'last_letter': name[-1],
                'last_three': name[-3:],
                'last_two': name[-2:]
            })

    def data_sets(self):
        """
        Raw data processing
        """
        
        with open('corpora/name_list.txt', 'r') as f:
            names = []
            for name_results in f:
                names.append(tuple(name_results.strip().split(',')))

        return names

    
    def feature_set(self):
        """
        Shuffling feature data sets
        """
        
        feature_sets = []

        for name_results in self.data_sets():
            name, gender = name_results

            name = self.gender_features(name)
            feature_sets.append((name, gender))

        return shuffle(feature_sets)

    
    def model_train(self, percent_to_train=0.70):
        """
        Model training
        """
        
        feature_sets = self.feature_set()
        partition = int(len(feature_sets) * percent_to_train) # data set partition
        train_data_set = feature_sets[:partition] # training data set
        test_data_set = feature_sets[partition:] # test data set
        self.classifier = NaiveBayesClassifier.train(train_data_set)
        
        print("Classifier accuracy: {:0.2%}".format(accuracy(self.classifier, test_data_set)))

        return self.classifier
        
    def informative_features(self, num_of_feature=25):
        """
        Show most informative features
        """
        
        return self.model_train().show_most_informative_features(num_of_feature)
        
    def gender_classifier(self, name = None):
        try:
            gender = self.classifier.classify(self.gender_features(name.upper()))
            if gender == 'M':
                return 'My gender predicting result for "{}" is male.'.format(name)
            if gender == 'F':
                return 'My gender predicting result for "{}" is female.'.format(name)
        except IndexError:
            return 'Please enter correct name!'
        except TypeError:
            return 'Argument missing!'


if __name__ == '__main__':
    gcf = GenderClassifier()
    gcf.model_train()
    # gcf.feature_set()
    print(gcf.gender_classifier('Porimol Chandro'))
    print(gcf.informative_features())