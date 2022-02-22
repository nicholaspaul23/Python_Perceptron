import random
from datetime import datetime

# Constants
INPUT_SIZE = 2
WEIGHT_SIZE = INPUT_SIZE + 1
LEARNING_RATE = 0.1
ITERATIONS = 10

# Globa vars
weights = [0] * WEIGHT_SIZE

# Initialize Network
def initializeNetwork():
  random.seed(datetime.now())
	
  # initialize weights with random values
  for x in range(WEIGHT_SIZE):
    global weights
    weights[x] = random.random()


# Feedforward Process
def feedForward(inputVector):
  sum = 0.0
  global weights

  # calculate inputs * weights
  for x in range(INPUT_SIZE):
    sum = sum + weights[x] * inputVector[x]
	
  # include bias	
  sum = sum + weights[WEIGHT_SIZE-1]
	
  # activation function (OR 1 if val >= 0)
  result = 1 if sum >= 1.0 else 0
	
  return result
	


# Train Network
def train():
  iterations = 0
  
  global weights

  trainData = [[0,0],[0,1],[1,0],[1,1]]
  
  while iterations < ITERATIONS:
    iterationError = 0.0

    print(f"Iterations: {iterations}\n")

    for x in range(len(trainData)):
      desiredOutput = trainData[x][0] or trainData[x][1]
      output = feedForward(trainData[x])

      error = desiredOutput - output

      print(f"{trainData[x][0]} OR {trainData[x][1]} = {output} ({desiredOutput})\n")

      weights[0] = weights[0] + (LEARNING_RATE * float(error) * float(trainData[x][0]))
      weights[1] = weights[1] + (LEARNING_RATE * float(error) * float(trainData[x][1]))
      weights[2] = weights[2] + (LEARNING_RATE * float(error))

      iterationError = iterationError + (error * error)

    iterations = iterations + 1

    print(f"Iteration error {iterationError}\n\n")

    if(iterationError == 0.0):
      break

  #end-while-loop




# Driver Code
try:
  initializeNetwork()
  train()
  print(f"Final weights {weights[0]} {weights[1]} bias {weights[2]}")
except:
  print(f"Error running code...\n")
