// helloworld.cpp
#include <iostream>
#include <string>

// 1) Print a greeting
void greet(const std::string &name) {
    std::cout << "Hello, " << name << "!" << std::endl;
}

// 2) Add two integers and return the result
int add(int a, int b) {
    return a + b;
}

// 3) Multiply two integers and return the result
int multiply(int a, int b) {
    return a * b;
}

int main() {
    greet("Alice");

    int x = 3, y = 5;
    std::cout << "add(" << x << ", " << y << ") = " << add(x, y) << std::endl;
    std::cout << "multiply(" << x << ", " << y << ") = " << multiply(x, y) << std::endl;

    return 0;
}