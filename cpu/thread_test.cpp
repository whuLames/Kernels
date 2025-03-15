#include <iostream>
#include <thread>
#include <vector>
#include <memory>
using namespace std;

void func(shared_ptr<vector<int>> val,  int a)
{
    // (*vals)[a] = a;
    (*val)[a] = a;
}

void fun(int& a)
{
    for(int i = 0; i < 100; i ++) a = a + 1;
}

int main()
{
    int n;
    cin >> n;
    thread x[n];
    // vector<int> vals(5, 0);
    // shared_ptr<vector<int>> vec(&vals, [](vector<int>*) {
    //     cout << "custom delete" << endl;
    // });
    int a = 0;
    for(int i = 0; i < n; i ++) x[i] = thread(fun, ref(a));

    for(int i = 0; i < n; i ++) x[i].join();

    cout << a << endl;
    // for(int i = 0; i < 5; i ++) cout << vals[i] << endl;
}