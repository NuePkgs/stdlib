# NEXT .PKG IDEA

```python
import math
import random

class Matrix:
    def __init__(self, rows, cols, data=None):
        if rows < 2 or cols < 2 or rows > 128 or cols > 128:
            raise ValueError("Matrix dimensions must be between 2 and 128")
        
        self.rows = rows
        self.cols = cols
        
        if data is None:
            self.data = [[0 for _ in range(cols)] for _ in range(rows)]
        else:
            if len(data) != rows or len(data[0]) != cols:
                raise ValueError("Data dimensions do not match specified matrix size")
            self.data = [row[:] for row in data]  # Deep copy of data

    def to_string(self):
        return "\n".join([" ".join([f"{x:8.3f}" for x in row]) for row in self.data])

    def add(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for addition")
        result = [[self.data[i][j] + other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(self.rows, self.cols, result)

    def subtract(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for subtraction")
        result = [[self.data[i][j] - other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(self.rows, self.cols, result)

    def multiply(self, other):
        if isinstance(other, (int, float)):
            result = [[self.data[i][j] * other for j in range(self.cols)] for i in range(self.rows)]
            return Matrix(self.rows, self.cols, result)
        elif isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("Number of columns in first matrix must equal number of rows in second matrix")
            result = [[sum(self.data[i][k] * other.data[k][j] for k in range(self.cols)) 
                       for j in range(other.cols)] for i in range(self.rows)]
            return Matrix(self.rows, other.cols, result)

    def power(self, power):
        if self.rows != self.cols:
            raise ValueError("Matrix must be square for power operation")
        if not isinstance(power, int) or power < 0:
            raise ValueError("Power must be a non-negative integer")
        result = Matrix.identity(self.rows)
        temp = self
        while power > 0:
            if power % 2 == 1:
                result = result.multiply(temp)
            temp = temp.multiply(temp)
            power //= 2
        return result

    def transpose(self):
        result = [[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)]
        return Matrix(self.cols, self.rows, result)

    def determinant(self):
        if self.rows != self.cols:
            raise ValueError("Determinant can only be calculated for square matrices")
        if self.rows == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            submatrix = [row[:j] + row[j+1:] for row in self.data[1:]]
            det += (-1) ** j * self.data[0][j] * Matrix(self.rows-1, self.cols-1, submatrix).determinant()
        return det

    @staticmethod
    def identity(size):
        if size < 2 or size > 128:
            raise ValueError("Matrix size must be between 2 and 128")
        return Matrix(size, size, [[1 if i == j else 0 for j in range(size)] for i in range(size)])

    def inverse(self):
        if self.rows != self.cols:
            raise ValueError("Only square matrices can be inverted")
        det = self.determinant()
        if abs(det) < 1e-10:
            raise ValueError("Matrix is not invertible")
        if self.rows == 2:
            a, b, c, d = self.data[0][0], self.data[0][1], self.data[1][0], self.data[1][1]
            return Matrix(2, 2, [[d/det, -b/det], [-c/det, a/det]])
        adj = self.adjugate()
        return adj.multiply(1/det)

    def adjugate(self):
        cofactors = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                submatrix = [row[:j] + row[j+1:] for row in (self.data[:i] + self.data[i+1:])]
                cofactors[i][j] = (-1) ** (i+j) * Matrix(self.rows-1, self.cols-1, submatrix).determinant()
        return Matrix(self.rows, self.cols, cofactors).transpose()

    def trace(self):
        if self.rows != self.cols:
            raise ValueError("Trace can only be calculated for square matrices")
        return sum(self.data[i][i] for i in range(self.rows))

    def rank(self):
        epsilon = 1e-10
        ref = self.to_row_echelon_form()
        rank = 0
        for i in range(min(self.rows, self.cols)):
            if abs(ref.data[i][i]) > epsilon:
                rank += 1
        return rank

    def to_row_echelon_form(self):
        result = [row[:] for row in self.data]  # Make a copy
        lead = 0
        for r in range(self.rows):
            if lead >= self.cols:
                return Matrix(self.rows, self.cols, result)
            i = r
            while result[i][lead] == 0:
                i += 1
                if i == self.rows:
                    i = r
                    lead += 1
                    if self.cols == lead:
                        return Matrix(self.rows, self.cols, result)
            result[i], result[r] = result[r], result[i]
            lv = result[r][lead]
            result[r] = [mrx / float(lv) for mrx in result[r]]
            for i in range(self.rows):
                if i != r:
                    lv = result[i][lead]
                    result[i] = [iv - lv*rv for rv, iv in zip(result[r], result[i])]
            lead += 1
        return Matrix(self.rows, self.cols, result)

    def is_symmetric(self):
        if self.rows != self.cols:
            return False
        return all(self.data[i][j] == self.data[j][i] for i in range(self.rows) for j in range(i+1, self.cols))

    def is_orthogonal(self):
        if self.rows != self.cols:
            return False
        identity = Matrix.identity(self.rows)
        return (self.multiply(self.transpose()).is_close(identity) and 
                self.transpose().multiply(self).is_close(identity))

    def is_close(self, other, rel_tol=1e-09, abs_tol=0.0):
        if self.rows != other.rows or self.cols != other.cols:
            return False
        return all(math.isclose(self.data[i][j], other.data[i][j], rel_tol=rel_tol, abs_tol=abs_tol)
                   for i in range(self.rows) for j in range(self.cols))

    def frobenius_norm(self):
        return math.sqrt(sum(x*x for row in self.data for x in row))

    @staticmethod
    def random(rows, cols, low=0, high=1):
        if rows < 2 or cols < 2 or rows > 128 or cols > 128:
            raise ValueError("Matrix dimensions must be between 2 and 128")
        return Matrix(rows, cols, [[random.uniform(low, high) for _ in range(cols)] for _ in range(rows)])

    def reshape(self, new_rows, new_cols):
        if new_rows * new_cols != self.rows * self.cols:
            raise ValueError("New dimensions must have the same number of elements as the original matrix")
        flat = [x for row in self.data for x in row]
        reshaped = [flat[i:i+new_cols] for i in range(0, len(flat), new_cols)]
        return Matrix(new_rows, new_cols, reshaped)

    def to_list(self):
        return [row[:] for row in self.data]

    @classmethod
    def from_list(cls, data):
        rows = len(data)
        cols = len(data[0]) if rows > 0 else 0
        return cls(rows, cols, data)

    def apply_function(self, func):
        return Matrix(self.rows, self.cols, [[func(x) for x in row] for row in self.data])
```

## usage

```python
# Create a 4x4 matrix
A = Matrix(4, 4, [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

# Create another 4x4 matrix
B = Matrix(4, 4, [
    [16, 15, 14, 13],
    [12, 11, 10, 9],
    [8, 7, 6, 5],
    [4, 3, 2, 1]
])

# Addition
C = A.add(B)
print("A + B:")
print(C.to_string())

# Subtraction
D = A.subtract(B)
print("\nA - B:")
print(D.to_string())

# Multiplication
E = A.multiply(B)
print("\nA * B:")
print(E.to_string())

# Transpose
AT = A.transpose()
print("\nTranspose of A:")
print(AT.to_string())

# Determinant
det_A = A.determinant()
print(f"\nDeterminant of A: {det_A}")

# Inverse (if possible)
try:
    A_inv = A.inverse()
    print("\nInverse of A:")
    print(A_inv.to_string())
except ValueError as e:
    print(f"\nCould not calculate inverse: {e}")

# Create a larger random matrix (10x10)
R = Matrix.random(10, 10, 0, 100)
print("\nRandom 10x10 matrix:")
print(R.to_string())

# Calculate Frobenius norm
norm_R = R.frobenius_norm()
print(f"\nFrobenius norm of R: {norm_R}")

# Check if R is symmetric
is_sym = R.is_symmetric()
print(f"\nIs R symmetric? {is_sym}")

# Create identity matrix
I = Matrix.identity(5)
print("\n5x5 Identity matrix:")
print(I.to_string())

# Matrix power
A_cubed = A.power(3)
print("\nA^3:")
print(A_cubed.to_string())

# Reshape matrix
reshaped = A.reshape(2, 8)
print("\nA reshaped to 2x8:")
print(reshaped.to_string())

# Apply function (e.g., square each element)
squared = A.apply_function(lambda x: x**2)
print("\nA with each element squared:")
print(squared.to_string())

# Create a matrix from a list
list_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
F = Matrix.from_list(list_data)
print("\nMatrix created from list:")
print(F.to_string())

# Calculate rank
rank_F = F.rank()
print(f"\nRank of F: {rank_F}")

# Trace
trace_F = F.trace()
print(f"\nTrace of F: {trace_F}")
```
