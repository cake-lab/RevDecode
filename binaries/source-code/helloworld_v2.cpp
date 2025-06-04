// helloworld_v2.cpp
#include <iostream>
#include <string>

// 1) Slightly different greeting text
void greet(const std::string &name) {
    std::cout << "Hey there, " << name << "! Welcome!" << std::endl;
}

// 2) Add two integers, but swap the order of operations in the body
int add(int a, int b) {
    int sum = b + a;  // swapped order, but result is identical
    return sum;
}

// 3) Multiply, but store intermediate in a variable
int multiply(int a, int b) {
    int product = a * b;
    return product;
}

int main() {
    greet("Alice");

    int x = 3, y = 5;
    std::cout << "add(" << x << ", " << y << ") = " << add(x, y) << std::endl;
    std::cout << "multiply(" << x << ", " << y << ") = " << multiply(x, y) << std::endl;

    return 0;
}