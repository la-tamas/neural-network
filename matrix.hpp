#pragma once

#ifndef MATRIX_HPP_INCLUDED
#define MATRIX_HPP_INCLUDED

#define assert(cond)             \
  do {                           \
    if (!(cond)) __debugbreak(); \
  } while (false)

#include <vector>
#include <math.h>

typedef float matrix_t;

class NN_Matrix {
public:
    NN_Matrix(int rows = 0, int cols= 0, matrix_t val = 0);
    NN_Matrix& init(int rows, int cols, matrix_t val = 0);

    matrix_t at(int row, int col) const;
    void set(int row, int col, matrix_t value);

    std::vector<matrix_t>& data();
    const std::vector<matrix_t>& data() const;

    void print() const;

    int indexOfMax();
    int rows() const;
    int cols() const;

    matrix_t sum() const;
    NN_Matrix& randomize(matrix_t min = 0, matrix_t max = 1);
    NN_Matrix& sigmoid();
    NN_Matrix& square();
    NN_Matrix transpose() const;
    NN_Matrix multiply(const NN_Matrix& other) const;
    NN_Matrix& multiply_inplace(const NN_Matrix& other); // Element by element.

    // Operators
    NN_Matrix& operator+=(const NN_Matrix& other);

    NN_Matrix operator-(const NN_Matrix& other) const;
    NN_Matrix operator*(const NN_Matrix& other) const;
    NN_Matrix operator*(matrix_t value) const;
private:
    int _rows, _cols;
    std::vector<matrix_t> _data;
};

NN_Matrix::NN_Matrix(int rows, int cols, matrix_t val) : _rows(rows), _cols(cols), _data(rows * cols, val)
{}

NN_Matrix& NN_Matrix::init(int rows, int cols, matrix_t val) {
    this->_rows = rows;
    this->_cols = cols;
    this->_data = std::vector<matrix_t>(rows * cols, val);
    return *this;
}

NN_Matrix& NN_Matrix::randomize(matrix_t min, matrix_t max) {

    assert(max > min);
    for (size_t i = 0; i < _data.size(); i++) {
        matrix_t val = (matrix_t)((float)rand() / (float)RAND_MAX) * (max - min) + min;
        _data[i] = val;
    }
    return *this;
}

std::vector<matrix_t>& NN_Matrix::data() {
  return _data;
}

const std::vector<matrix_t>& NN_Matrix::data() const {
  return _data;
}

void NN_Matrix::print() const {
    printf("[\n");
    for (int r = 0; r < _rows; r++) {
        printf("  ");
        for (int c = 0; c < _cols; c++) {
            if (c != 0) printf(", ");
            // negative number has extra '-' character at the start.
            matrix_t val = at(r, c);
            if (val >= 0) printf(" %.6f", val);
            else printf("%.6f", val);
        }
        printf("\n");
    }
    printf("]\n");
}

int NN_Matrix::indexOfMax() {
    matrix_t maximum = 0;
    int index = 0;

    for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _cols; j++) {
            if (_data[i, j] > maximum) {
                maximum = _data[i, j];
                index = i + j;
            }
        }
    }

    return index;
}

int NN_Matrix::rows() const {
    return _rows;
}

int NN_Matrix::cols() const {
    return _cols;
}

matrix_t NN_Matrix::at(int row, int col) const {
  return _data[row * _cols + col];
}

void NN_Matrix::set(int row, int col, matrix_t value) {
    _data[row * _cols + col] = value;
}

matrix_t NN_Matrix::sum() const {
    matrix_t total = 0;
    for (size_t i = 0; i < _data.size(); i++) {
        total += _data[i];
    }
    return total;
}

static inline matrix_t sigmoid(matrix_t x) {
  return 1.f / (1.f + expf(-x));
}

NN_Matrix& NN_Matrix::sigmoid() {
  for (size_t i = 0; i < _data.size(); i++) {
    _data[i] = ::sigmoid(_data[i]);
  }
  return *this;
}

NN_Matrix& NN_Matrix::square() {
    for (size_t i = 0; i < _data.size(); i++) {
        _data[i] = _data[i] * _data[i];
    }
    return *this;
}

NN_Matrix NN_Matrix::transpose() const {
    NN_Matrix m(_cols, _rows);
    for (int r = 0; r < _rows; r++) {
        for (int c = 0; c < _cols; c++) {
            m.set(c, r, at(r, c));
        }
    }
    return m;
}

NN_Matrix NN_Matrix::multiply(const NN_Matrix& other) const {
    assert(_rows == other._rows && _cols == other._cols);
    NN_Matrix m(_rows, _cols);
    for (size_t i = 0; i < _data.size(); i++) {
        m._data[i] = _data[i] * other._data[i];
    }
    return m;
}

NN_Matrix& NN_Matrix::multiply_inplace(const NN_Matrix& other) {
    assert(_rows == other._rows && _cols == other._cols);
    for (size_t i = 0; i < _data.size(); i++) {
        _data[i] *= other._data[i];
    }
    return *this;
}

// Operators
NN_Matrix& NN_Matrix::operator+=(const NN_Matrix& other) {
    bool cond = (_rows == other._rows && _cols == other._cols);
    assert(_rows == other._rows && _cols == other._cols);
    for (size_t i = 0; i < _data.size(); i++) {
        _data[i] += other._data[i];
    }
    return *this;
}

NN_Matrix NN_Matrix::operator-(const NN_Matrix& other) const {
    assert(_rows == other._rows && _cols == other._cols);
    NN_Matrix m(this->_rows, this->_cols);
    for (size_t i = 0; i < _data.size(); i++) {
        m._data[i] = this->_data[i] - other._data[i];
    }
    return m;
}

NN_Matrix NN_Matrix::operator*(const NN_Matrix& other) const {

  // (r1 x c1) * (r2 x c2) =>
  //   assert(c1 == r2), result = (r1 x c2)
  assert(this->_cols == other._rows);

  NN_Matrix m(this->_rows, other._cols);

  int n = _cols; // Width or a row.
  for (int r = 0; r < m._rows; r++) {
    for (int c = 0; c < m._cols; c++) {

      matrix_t val = 0;
      for (int i = 0; i < n; i++) {
        val += this->at(r, i) * other.at(i, c);
      }
      m.set(r, c, val);
    }
  }

  return m;
}

NN_Matrix NN_Matrix::operator*(matrix_t value) const {
  NN_Matrix m(_rows, _cols);
  std::vector<matrix_t>& m_data = m.data();
  for (size_t i = 0; i < _data.size(); i++) {
    m_data[i] = _data[i] * value;
  }
  return m;
}


#endif // MATRIX_HPP_INCLUDED
