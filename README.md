# Creating a Virtual Environment with Python 3.10.6 using Conda (Windows)

This guide will walk you through the process of creating a virtual environment with Python version 3.10.6 using Conda on Windows.

## Prerequisites

- Anaconda or Miniconda should be installed on your Windows system. If you don't have it, you can download and install it from the official Anaconda website: https://www.anaconda.com/products/individual

## Instructions

1. Open the Anaconda Prompt from the Start menu. It is a command prompt specifically for Conda.

2. Update Conda (optional but recommended):

```
conda update conda
```

3. Create a new virtual environment with Python 3.10.6:

```
conda create --name myenv python=3.10.6
```

Replace `myenv` with the desired name for your environment. This command will create a new environment called `myenv` and install Python version 3.10.6 in it.

4. Activate the newly created environment:

```
conda activate myenv
```

Replace `myenv` with the name of your environment. Activating the environment ensures that any subsequent package installations or commands are executed within the virtual environment.

5. Install packages from a `requirements.txt` file:


```
pip install -r requirements.txt
```

Make sure the `requirements.txt` file is present in the current directory. This command will install all the packages listed in the file, along with their specified versions, into the virtual environment.

6. Verify the package installation:

```
pip list
```

This command will display the list of installed packages in the virtual environment. Make sure the required packages from `requirements.txt` are listed.

Now you have a virtual environment named `myenv` with Python version 3.10.6 set up using Conda on Windows. You have also installed the required packages from the `requirements.txt` file into the virtual environment.

Remember to activate the environment each time you want to work within it, using the command mentioned in step 4.
