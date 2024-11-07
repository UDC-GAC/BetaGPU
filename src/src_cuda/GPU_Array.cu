#include "GPU_Array.hpp"

/* --------------- GPU_Array Class  --------------- */

template <typename T>
GPU_Array<T>::GPU_Array(): size(0), data_d(nullptr), data_h(nullptr), memory_state(MemoryState::IN_SYNC) {}

template <typename T>
GPU_Array<T>::GPU_Array(size_t size): size(size), memory_state(MemoryState::IN_SYNC){
    this->allocate(size);
}

template <typename T>
GPU_Array<T>::GPU_Array(const T *data, size_t size): size(size), memory_state(MemoryState::IN_SYNC){
    this->allocate(size);
    cudaMemcpy(this->data_h, data, size * sizeof(T), cudaMemcpyHostToHost);
    cudaMemcpy(this->data_d, data, size * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
GPU_Array<T>::~GPU_Array(){
    if (this->data_d != nullptr)
        cudaFree(this->data_d);
    if (this->data_h != nullptr)
        cudaFreeHost(this->data_h);
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
void GPU_Array<T>::set_device_data(const T *data, size_t size, bool copy){
    cudaMemcpy(this->data_d, data, size * sizeof(T), cudaMemcpyDeviceToDevice);
    this->memory_state = MemoryState::DEVICE_ONLY;

    if (copy){
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
void GPU_Array<T>::set_host_data(const T *data, size_t size, bool copy){
    cudaMemcpy(this->data_h, data, size * sizeof(T), cudaMemcpyHostToHost);
    this->memory_state = MemoryState::HOST_ONLY;

    if (copy){
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
void GPU_Array<T>::allocate(size_t elements){
    if (this -> data_d != nullptr){
        cudaFree(this->data_d);
    }
    if (this -> data_h != nullptr){
        cudaFreeHost(this->data_h);
    }
    cudaMalloc(&this->data_d, elements * sizeof(T));
    cudaMallocHost(&this->data_h, elements * sizeof(T));
}

template class GPU_Array<int>;
template class GPU_Array<float>;
template class GPU_Array<double>;
