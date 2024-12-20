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
    
    // Rule of 5
    ~GPU_Array();
    GPU_Array(const GPU_Array &other); // Copy constructor
    GPU_Array(GPU_Array &&other) noexcept; // Move constructor
    GPU_Array &operator=(const GPU_Array &other); // Copy assignment
    GPU_Array &operator=(GPU_Array &&other) noexcept; // Move assignment



    const T *get_device_data();
    T *device_data(bool require_updated=true);
    void set_device_data(const T *data, size_t size, bool copy_to_host=true);

    const T *get_host_data();
    T *host_data(bool require_updated=true);
    void set_host_data(const T *data, size_t size, bool copy_to_device=true);

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
    void free();

};