#pragma once
#include <iostream>
#include <chrono>
#include <string>
class Timer
{
public:
    Timer() = default;
    ~Timer() = default;
    void time(std::string info = "", int count = 1)
    {
        if(info == ""){
            t = std::chrono::steady_clock::now();
        } else {
            auto t1 = std::chrono::steady_clock::now();
            float time = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t).count()) / 1000.0f / count;
            std::cout << "TimerInfo: " << info << " time: " << time << " ms." << std::endl;
            t = t1;
        }
    }

private:
    std::chrono::_V2::steady_clock::time_point t;
};

static Timer t;
