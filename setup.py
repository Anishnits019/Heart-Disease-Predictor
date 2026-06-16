from setuptools import find_packages, setup 
HYPHEN_DOT='-e.'
def get_requirements(file_path):
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if HYPHEN_DOT in requirements:
         requirements.remove(HYPHEN_DOT)
    return requirements


print(get_requirements('/Users/anish/Documents/Student Ml/requirements.txt'))
setup(
    name="Student ML",
    version="0.0.1",
    author='Anish',
    install_requires=get_requirements('requirements.txt')
)