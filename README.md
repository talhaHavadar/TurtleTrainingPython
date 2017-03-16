## TurtleTraining (Python)

This script collects data for training the neural network of Turtle Project (self-driving car thesis). To detect keyboard events, I used `PyGame`. There is no gui except `OpenCV`'s itself.

### Dependencies
- OpenCV v3.0.0 or above (tested in OpenCV v3.2.0)
- NumPy (this is also dependency of OpenCV)

### Directions Matrix
In my case, I have 5 possible directions to evaluate. Theese are:
- Full Left (45 degree) `[1, 0, 0, 0, 0]`
- Left (22.5 degree) `[0, 1, 0, 0, 0]`
- Forward `[0, 0, 1, 0, 0]`
- Right (-22.5 degree) `[0, 0, 0, 1, 0]`
- Full Right (-45 degree) `[0, 0, 0, 0, 1]`

`I assumed left side is positive vector and right side is negative vector.`
So according to these assumptions, I created a matrix like below:
```python
self.directions = np.zeros((5,5))
for i in range(5):
    self.directions[i, i] = 1

"""
    [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ]
"""
```
