#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <shared_mutex>
#include <condition_variable>
#include <chrono>
#include <mutex>

template <typename T>
class TSStack
{
    std::shared_timed_mutex _lock;
    std::condition_variable_any _pushed;
    std::vector<T> _elements;
public:
    void push(T element)
    {
        std::unique_lock<std::shared_timed_mutex> writeLocker(_lock);
        _elements.push_back(element);
        _pushed.notify_all();
    }

    T pop()
    {
        std::unique_lock<std::shared_timed_mutex> locker(_lock);
        if(_elements.empty())
            _pushed.wait(locker,[&](){return !_elements.empty();});
        T a = _elements.back();
        _elements.pop_back();
        return a;
    }
    T top()
    {
        std::shared_lock<std::shared_timed_mutex> readLocker(_lock);
        if(_elements.empty())
            _pushed.wait(readLocker,[&](){return !_elements.empty();});
        return _elements.back();
    }
    int divs(T elem) {
        std::unique_lock<std::shared_timed_mutex> writeLocker(_lock);
        int result =
                count_if(_elements.begin(), _elements.end(), [&elem] (int _n)
                {
                    if(_n == 0) throw(std::exception());
                    return (elem % _n) == 0;
                });
        return result;
    }

};

void testPushTS(TSStack<int> &a, int b, int f){
    for(int i = 0; i<b; i++){
        a.push(i+f);
        a.push(f+i+1);
        std::cout << a.pop() << std::endl;
    }
}
void testPush(TSStack<int> &a, int b){
    for(int i = 0; i<b; i++){
        a.push(i);
    }
}
void testPop(TSStack<int> &a, int b){
    for(int i = 0; i<b; i++)
        std::cout << a.pop() << std::endl;
}


void testSleepPush(TSStack<int> &a, int n){
    for(int i = 0; i<2; i++) {
        for (int j = 0; j < n; j++)
            a.push(i);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}
void testWaitTop(TSStack<int> &a, int n){
    for (int i = 0; i < n; ++i) {
        a.pop();
    }
    std::cout<<"done"<<std::endl;
}
void testInside(TSStack<int> &a, int n){
    std::cout<<"trying"<<std::endl;
    //std::cout<<a.divs(n)<<std::endl;
    a.divs(n);
}

int main() {
    int threadsNum = 5;
    TSStack<int> myStack;
    std::vector<std::thread> threads;
    for(int i = 0; i<threadsNum; i++){
        threads.emplace_back(std::thread(testPushTS, std::ref(myStack), 1,i));
    }
//    for(int i = 0; i<threadsNum; i++){
//        threads.emplace_back(std::thread(testPop, std::ref(myStack), 1));
//    }
    for(std::thread &thread: threads)
        if(thread.joinable())
            thread.join();
    std::cout<< "content" <<std::endl;
    std::thread t(testPop, std::ref(myStack), 5);
    t.join();


    std::cout<< "waitTest" <<std::endl;
    std::thread sendT(testSleepPush, std::ref(myStack), 4);
    std::thread recvT(testWaitTop, std::ref(myStack), 5);
    sendT.join();
    recvT.join();


    std::cout<< "throwTest" <<std::endl;
    std::thread brk(testPush, std::ref(myStack), 1);
    std::thread e(testInside, std::ref(myStack), 4);
    std::thread fix(testPop, std::ref(myStack), 1);
    std::thread noE(testInside, std::ref(myStack), 4);
    brk.join();
    e.join();
    fix.join();
    noE.join();

    //return 0;
    //std::sort()
}
