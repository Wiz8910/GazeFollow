from setuptools import setup

setup(name='GazeFollow',
      version='0.2',
      description='Predict gaze of subject in image using various computer vision methods.',
      url='https://github.com/Wiz8910/GazeFollow',
      author='Adam Bowers, Colin Conduff',
      author_email='colin.conduff@mst.edu',
      license='MIT',
      packages=['GazeFollow'],
      scripts=['GazeFollow/gazefollow'],
      install_requires=[
          'tensorflow',
          'scipy',
          'numpy',
          'pillow',
          'matplotlib',
          'sklearn'
      ],
      include_package_data=True,
      zip_safe=False)
