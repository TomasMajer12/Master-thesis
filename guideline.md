Design a method for training a Sudoku puzzle solver from examples of puzzle assignments and their corresponding solutions. The solver should support both symbolic inputs (numerical grid representations) and visual inputs (images of Sudoku puzzles). The core of the solver will be a neural network whose final decision layer is a Markov Network predictor. An efficient learning algorithm for this prediction model must be designed and implemented. The resulting system will be empirically evaluated and compared against suitable baseline approaches. Requirements 

• Become familiar with methods for learning neural networks and Markov networks. 
• Design and implement a Python library for training and evaluating the proposed predictors. 
• Create a benchmark dataset of Sudoku puzzles with ground-truth solutions, covering both symbolic and visual inputs. 
• Implement straightforward baseline solutions, for example: a two-stage system consisting of single-digit OCR followed by a hard-coded Sudoku solver. 
• Apply the proposed system to the Sudoku benchmark and perform a statistical evaluation of the results, including comparison with baseline methods.
