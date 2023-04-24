from setuptools import setup,find_packages
from typing import List


HYPON_E_DOT = "-e ."

def get_requirements(file_path:str)->List[str]:
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.replace("\n","") for i in requirements]

        if HYPON_E_DOT in requirements:
            requirements.remove(HYPON_E_DOT)

    return requirements


setup(
    name="BigMarketSales_Prediction",
    version="0.1",
    author="yash mohite",
    author_email="mohite,yassh@gmial.com",
    packages=find_packages(),
    install_require=get_requirements("requirements.txt")
)