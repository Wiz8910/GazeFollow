from setuptools import setup

setup(name='GazeFollow',
      version='0.1',
      description='Predict gaze of subject in image using various computer vision methods.',
      url='https://github.com/Wiz8910/GazeFollow',
      author='Adam Bowers, Colin Conduff',
      author_email='colin.conduff@mst.edu',
      license='MIT',
      packages=['GazeFollow'],
      install_requires=[
          'tensorflow',
          'scipy',
          'numpy',
          'pillow',
          'matplotlib',
          'sklearn'
      ],
      scripts=['bin/gaze_follow'],
      include_package_data=True,
      zip_safe=False)
