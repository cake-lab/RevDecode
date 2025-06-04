// helloworld_v4.cpp
#include <iostream>
#include <string>

// 1) Greeting uses a different function signature (no const & but copies)
void greet(std::string name) {
    std::cout << "Greetings, " << name << "." << std::endl;
}

// 2) Add takes its integers by reference (const) instead of by value
int add(const int &a, const int &b) {
    return a + b;
}

// 3) Multiply returns a `long` instead of `int`, and uses 64-bit multiplication
long multiply(int a, int b) {
    long A = static_cast<long>(a);
    long B = static_cast<long>(b);
    return A * B;
}

int main() {
    greet("Alice");

    int x = 3, y = 5;
    std::cout << "add(" << x << ", " << y << ") = " << add(x, y) << std::endl;
    std::cout << "multiply(" << x << ", " << y << ") = " << multiply(x, y) << std::endl;

    return 0;
}