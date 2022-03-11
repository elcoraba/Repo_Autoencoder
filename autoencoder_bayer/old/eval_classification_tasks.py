from scipy import io


class Biometrics():
    def get_zl(self, data):
        _bool = data.corpus.apply(
            lambda x: x in ['EMVIC2014', 'Cerf2007-FIFA', 'ETRA2019'])
        data = data[_bool]
        data = data[~data['subj'].str.contains('test')]
        return data.z, data.subj

class AgeGroupBinary():
    def get_zl(self, data):
        def bin(s):
            age = ages[s]
            if age in range(18, 23):
                return '18-22'
            return '23-35'

        data = data[data.corpus == 'Cerf2007-FIFA']
        subjects = io.loadmat(
            'data/Cerf2007-FIFA/general', squeeze_me=True)['subject'] # ../
        ages = {s[4]: s[2] for s in subjects}
        # [data corpus subj stim in_vel z_vel z]
        print('data ', data[0])
        print('AgeGroupBinary ', data.z[0])
        return data.z, data['subj'].apply(lambda x: bin(x))
        # z
        # 0       [0.009964153, -0.006462234, 0.004003319, 0.107...
        # 1       [0.0090957945, -0.006462234, 0.0056486833, 0.1...

class GenderBinary():
    def get_zl(self, data):
        data = data[data.corpus == 'Cerf2007-FIFA']
        subjects = io.loadmat(
            'data/Cerf2007-FIFA/general', squeeze_me=True)['subject'] # ../
        legend = {1: 'male', 0: 'female'}
        subj_sexes = {s[4]: legend[s[0]] for s in subjects}
        return data.z, data['subj'].apply(lambda x: subj_sexes[x])


TASKS = {
    'age-group': AgeGroupBinary,
    #'gender': GenderBinary
}
