#include "GPU_Array.hpp"

/* --------------- GPU_Array Class  --------------- */

template <typename T>
GPU_Array<T>::GPU_Array(): size(0), data_d(nullptr), data_h(nullptr), memory_state(MemoryState::IN_SYNC) {}

template <typename T>
GPU_Array<T>::GPU_Array(size_t size): size(size), memory_state(MemoryState::IN_SYNC){
    this->allocate(size);
}

template <typename T>
GPU_Array<T>::GPU_Array(const T *data, size_t size): size(size), memory_state(MemoryState::IN_SYNC) {
    this->allocate(size);
    cudaMemcpy(this->data_h, data, size * sizeof(T), cudaMemcpyHostToHost);
    cudaMemcpy(this->data_d, data, size * sizeof(T), cudaMemcpyHostToDevice);
}

// Rule of 5

// I.-   Destructor
template <typename T>
GPU_Array<T>::~GPU_Array(){
    this->free();
}

// II.-  Copy constructor
template <typename T>
GPU_Array<T>::GPU_Array(const GPU_Array &other): size(other.size), memory_state(other.memory_state){
    this->allocate(other.size);
    if (other.memory_state == MemoryState::HOST_ONLY || other.memory_state == MemoryState::IN_SYNC)
        cudaMemcpy(this->data_h, other.data_h, other.size * sizeof(T), cudaMemcpyHostToHost);
    if (other.memory_state == MemoryState::DEVICE_ONLY || other.memory_state == MemoryState::IN_SYNC)
        cudaMemcpy(this->data_d, other.data_d, other.size * sizeof(T), cudaMemcpyDeviceToDevice);
}

// III.- Copy assignment
template <typename T>
GPU_Array<T> &GPU_Array<T>::operator=(const GPU_Array &other){

    if (this == &other)
        return *this;

    this->size = other.size;
    this->memory_state = other.memory_state;
    this->allocate(other.size);
    if (other.memory_state == MemoryState::HOST_ONLY || other.memory_state == MemoryState::IN_SYNC)
        cudaMemcpy(this->data_h, other.data_h, other.size * sizeof(T), cudaMemcpyHostToHost);
    if (other.memory_state == MemoryState::DEVICE_ONLY || other.memory_state == MemoryState::IN_SYNC)
        cudaMemcpy(this->data_d, other.data_d, other.size * sizeof(T), cudaMemcpyDeviceToDevice);

    return *this;
}

// IV.-  Move constructor
template <typename T>
GPU_Array<T>::GPU_Array(GPU_Array &&other) noexcept
    : size(other.size), data_d(std::exchange(other.data_d, nullptr)), 
      data_h(std::exchange(other.data_h, nullptr)), memory_state(other.memory_state){}

// V.-   Move assignment
template <typename T>
GPU_Array<T> &GPU_Array<T>::operator=(GPU_Array &&other) noexcept {
    if (this == &other)
        return *this;

    this->size = other.size;
    this->memory_state = other.memory_state;
    std::swap(this->data_d, other.data_d);
    tstd::swap(this->data_h, other.data_h);
    

    return *this;
}

template <typename T>
const T *GPU_Array<T>::get_device_data(){
    if (this->memory_state == MemoryState::HOST_ONLY)
        this->copy_to_device(this->size);

    return this->data_d;
}

template <typename T>
T *GPU_Array<T>::device_data(){
    if (this->memory_state == MemoryState::HOST_ONLY)
        this->copy_to_device(this->size);

    this->memory_state = MemoryState::DEVICE_ONLY;
    return this->data_d;
}

template <typename T>
void GPU_Array<T>::set_device_data(const T *data, size_t size, bool copy_to_host){
    cudaMemcpy(this->data_d, data, size * sizeof(T), cudaMemcpyDeviceToDevice);
    this->memory_state = MemoryState::DEVICE_ONLY;

    if (copy_to_host){
        this->copy_to_host(size);
        this->memory_state = MemoryState::IN_SYNC;
    }
}

template <typename T>
const T *GPU_Array<T>::get_host_data(){
    if (this->memory_state == MemoryState::DEVICE_ONLY)
        this->copy_to_host(this->size);

    return this->data_h;
}

template <typename T>
T *GPU_Array<T>::host_data(){
    if (this->memory_state == MemoryState::DEVICE_ONLY)
        this->copy_to_host(this->size);

    this->memory_state = MemoryState::HOST_ONLY;
    return this->data_h;
}

template <typename T>
void GPU_Array<T>::set_host_data(const T *data, size_t size, bool copy_to_device){
    cudaMemcpy(this->data_h, data, size * sizeof(T), cudaMemcpyHostToHost);
    this->memory_state = MemoryState::HOST_ONLY;

    if (copy_to_device){
        this->copy_to_device(size);
        this->memory_state = MemoryState::IN_SYNC;
    }
}

template <typename T>
size_t GPU_Array<T>::get_size(){
    return this->size;
} 

template <typename T>
MemoryState GPU_Array<T>::get_memory_state(){
    return this->memory_state;
}

/* --------------- Private Methods  --------------- */

template <typename T>
void GPU_Array<T>::copy_to_device(size_t elements){
    cudaMemcpy(this->data_d, this->data_h, elements* sizeof(T), cudaMemcpyHostToDevice);
    this->memory_state = MemoryState::IN_SYNC;
}

template <typename T>
void GPU_Array<T>::copy_to_host(size_t elements){
    cudaMemcpy(this->data_h, this->data_d, elements * sizeof(T), cudaMemcpyDeviceToHost);
    this->memory_state = MemoryState::IN_SYNC;
}

template <typename T>
void GPU_Array<T>::free(){
    if (this->data_d != nullptr)
        cudaFree(this->data_d);
    if (this->data_h != nullptr)
        cudaFreeHost(this->data_h);

    this->data_d = nullptr;
    this->data_h = nullptr;
}

template <typename T>
void GPU_Array<T>::allocate(size_t elements){
    this->free();
    cudaMalloc(&this->data_d, elements * sizeof(T));
    cudaMallocHost(&this->data_h, elements * sizeof(T));
    if (...){
        this->free();
        throw std::bad_alloc();
    }
}


template class GPU_Array<int>;
template class GPU_Array<float>;
template class GPU_Array<double>;
