#pragma once

#include <cstddef> // for size_t

enum class MemoryState {
    HOST_ONLY,
    DEVICE_ONLY,
    IN_SYNC   
};

template <typename T>
class GPU_Array { 
public:
    GPU_Array();
    GPU_Array(size_t size);
    GPU_Array(const T *data, size_t size);
    ~GPU_Array();


    const T *get_device_data();
    T *device_data();
    void set_device_data(const T *data, size_t size, bool copy=true);

    const T *get_host_data();
    T *host_data();
    void set_host_data(const T *data, size_t size, bool copy=true);

    size_t get_size();
    MemoryState get_memory_state();

private:
    size_t size;
    T *data_d;
    T *data_h;
    MemoryState memory_state;

private:
    void copy_to_device(size_t elements);
    void copy_to_host(size_t elements);
    void allocate(size_t elements);

};