from setuptools import setup, find_packages

setup(name='continuousSafetyGym',
      version='0.0.1',
      description='safe RL environments with continuous cost functions',
      url='https://github.com/felipperoza/continuous-safety-gym',
      author='Felippe Roza',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gymnasium>=0.26.3', 'numpy==1.23.5','PyYAML==6.0.1', 'pybullet>=3.0.6', 'imageio>=2.30.0', 'mujoco==2.3.3', 'xmltodict>=0.13.0']
)
