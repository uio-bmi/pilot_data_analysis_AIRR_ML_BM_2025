from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='pilot_data_analysis_AIRR_ML_BM_2025',
    version='1.0',
    packages=['pilot_data_analysis'],
    url='',
    license='MIT',
    author='Chakravarthi Kanduri',
    author_email='chakra.kanduri@gmail.com',
    description='',
    include_package_data=True,
    zip_safe=False,
    entry_points={'console_scripts': [
                                      'split_train_test_all_dirs=pilot_data_analysis.split_train_test'
                                      ':execute_on_multiple_dirs',
                                      'split_train_test_single_dir=pilot_data_analysis.split_train_test'
                                      ':execute_on_single_dir',
                                      'profile_ml=pilot_data_analysis.profile_ml:execute',
                                      'logistic_interpretability=pilot_data_analysis.logistic_interpretability:execute',
                                      'emerson_interpretability=pilot_data_analysis.emerson_interpretability:execute',
                                      ]
    }
)
