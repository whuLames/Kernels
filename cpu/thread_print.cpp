#include <iostream>
#include <thread>
#include <mutex>
#include <memory>
#include <atomic>
#include <condition_variable>
using namespace std;

// for printFunc
// std::atomic<int> cur_val(0);

// for printInturn_1
int cur_id = 0;
int cur_val = 0;

// for printInturn_2
int counter = 0;
int turn = 0;
condition_variable cv;

void printFunc(int tid)
{

    // while(cur_val.load() < 100) {
    //     int val = cur_val.load();
    //     printf("Thread %d is printing value: %d\n", tid, val + 1);
    //     //atomic_fetch_add(&cur_val, 1);
    //     cur_val += 1;
    // }
    printf("Thread %d working", tid);
    static mutex mtx;
    while (true)
    {
        {
            lock_guard<mutex> lock(mtx);
            if(cur_val >= 100) break;
            printf("Thread %d is printing value: %d\n", tid, cur_val + 1);
            cur_val ++;
            // if(cur_val >= 100) break;
        }
    }
    
}

void printInturn_1(int tid)
{
    printf("Thread %d working\n", tid);
    static mutex mtx;
    
    // 交替打印
    while(true) {
        if(tid == cur_id) {
            //printf("Thread %d working", tid);
            {
                lock_guard<mutex> lock(mtx);
                if(cur_val >= 100) {cur_id = (cur_id + 1) % 3; break;}
                printf("Thread %d print value %d\n", tid, cur_val + 1);
                cur_val ++;
                cur_id = (cur_id + 1) % 3;
            }
        }
    }

}

void printInturn_2(int tid)
{
    static mutex mtx;
    while(true) {
        unique_lock<mutex> lock(mtx);
        cv.wait(lock, [&](){return tid==turn || counter >= 100;});
        if(counter >= 100) break;
        printf("Thread %d print value %d \n", tid, counter + 1);
        counter ++;
        turn = (turn + 1) % 3;
        cv.notify_all();
        // lock.unlock(); 离开作用域自动释放
    }
   

}
int main()
{
    thread workThreads[3];
    for(int i = 0; i < 3; i ++) {
        workThreads[i] = thread(printInturn_2, i);
    }
    printf("Call all threads");
    // printf("Thread %d working", tid);
    for(int i = 0; i < 3; i ++) workThreads[i].join();

}