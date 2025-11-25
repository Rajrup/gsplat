#include "nvcomp_wrapper.h"
#include "timer.h"
#include <nvcomp.h>
#include <nvcomp/lz4.h>
#include <nvcomp/snappy.h>
#include <nvcomp/gdeflate.h>
#include <nvcomp/deflate.h>
#include <nvcomp/zstd.h>
#include <nvcomp/cascaded.h>
#include <nvcomp/bitcomp.h>
#include <nvcomp/ans.h>
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// CUDA Error Checking
#define checkCudaErrors(val) check_((val), #val, __FILE__, __LINE__)
template<typename T>
void check_(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
    }
}

std::string nvcomp_algorithm_name(nvcompAlgorithm algorithm) {
    switch (algorithm) {
        case nvcompAlgorithm::LZ4: return "LZ4";
        case nvcompAlgorithm::SNAPPY: return "Snappy";
        case nvcompAlgorithm::GDEFLATE: return "GDeflate";
        case nvcompAlgorithm::DEFLATE: return "Deflate";
        case nvcompAlgorithm::ZSTANDARD: return "zStandard";
        case nvcompAlgorithm::CASCADED: return "Cascaded";
        case nvcompAlgorithm::BITCOMP: return "Bitcomp";
        case nvcompAlgorithm::ANS: return "ANS";
        default: return "Unknown";
    }
}

std::vector<nvcompAlgorithm> get_all_nvcomp_algorithms() {
    return {
        nvcompAlgorithm::LZ4,
        nvcompAlgorithm::SNAPPY,
        nvcompAlgorithm::GDEFLATE,
        nvcompAlgorithm::DEFLATE,
        nvcompAlgorithm::ZSTANDARD,
        nvcompAlgorithm::CASCADED,
        nvcompAlgorithm::BITCOMP,
        nvcompAlgorithm::ANS
    };
}

size_t nvcomp_get_max_compressed_size(size_t input_size_bytes, nvcompAlgorithm algorithm) {
    switch (algorithm) {
        case nvcompAlgorithm::LZ4: {
            size_t max_size;
            nvcompStatus_t status = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
                input_size_bytes, nvcompBatchedLZ4CompressDefaultOpts, &max_size);
            return (status == nvcompSuccess) ? max_size : input_size_bytes * 2;
        }
        case nvcompAlgorithm::SNAPPY: {
            size_t max_size;
            nvcompStatus_t status = nvcompBatchedSnappyCompressGetMaxOutputChunkSize(
                input_size_bytes, nvcompBatchedSnappyCompressDefaultOpts, &max_size);
            return (status == nvcompSuccess) ? max_size : input_size_bytes * 2;
        }
        case nvcompAlgorithm::GDEFLATE: {
            size_t max_size;
            nvcompStatus_t status = nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(
                input_size_bytes, nvcompBatchedGdeflateCompressDefaultOpts, &max_size);
            return (status == nvcompSuccess) ? max_size : input_size_bytes * 2;
        }
        // For other algorithms, use conservative estimate
        default:
            return input_size_bytes * 2;
    }
}

nvcompCompressionResult compress_nvcomp(
    const uint8_t* d_input_data,
    size_t input_size_bytes,
    nvcompAlgorithm algorithm,
    uint8_t* d_output_data,
    size_t max_output_size_bytes)
{
    nvcompCompressionResult result;
    result.success = false;
    result.original_size_bytes = input_size_bytes;
    result.compressed_size_bytes = 0;
    
    StopWatch sw;
    cudaStream_t stream = cudaStreamDefault;
    
    // Validate output buffer size
    (void)max_output_size_bytes;  // Suppress unused parameter warning
    
    // Allocate device arrays for batched API (single chunk)
    thrust::device_vector<const void*> d_input_ptrs_vec(1);
    thrust::device_vector<size_t> d_input_sizes_vec(1);
    thrust::device_vector<void*> d_output_ptrs_vec(1);
    thrust::device_vector<size_t> d_output_sizes_vec(1);
    thrust::device_vector<nvcompStatus_t> d_statuses_vec(1);
    
    d_input_ptrs_vec[0] = d_input_data;
    d_input_sizes_vec[0] = input_size_bytes;
    d_output_ptrs_vec[0] = d_output_data;
    
    switch (algorithm) {
        case nvcompAlgorithm::LZ4: {
            // Get temp workspace size
            size_t temp_bytes;
            nvcompStatus_t status = nvcompBatchedLZ4CompressGetTempSizeAsync(
                1, input_size_bytes, nvcompBatchedLZ4CompressDefaultOpts,
                &temp_bytes, input_size_bytes);
            if (status != nvcompSuccess) {
                result.error_message = "Failed to get LZ4 temp size: " + std::to_string(status);
                return result;
            }
            
            // Allocate temp workspace
            void* d_temp;
            checkCudaErrors(cudaMalloc(&d_temp, temp_bytes));
            
            // Compress
            status = nvcompBatchedLZ4CompressAsync(
                thrust::raw_pointer_cast(d_input_ptrs_vec.data()),
                thrust::raw_pointer_cast(d_input_sizes_vec.data()),
                input_size_bytes,
                1,
                d_temp,
                temp_bytes,
                thrust::raw_pointer_cast(d_output_ptrs_vec.data()),
                thrust::raw_pointer_cast(d_output_sizes_vec.data()),
                nvcompBatchedLZ4CompressDefaultOpts,
                thrust::raw_pointer_cast(d_statuses_vec.data()),
                stream);
            
            checkCudaErrors(cudaStreamSynchronize(stream));
            checkCudaErrors(cudaFree(d_temp));
            
            // Copy result back
            thrust::host_vector<size_t> h_output_sizes = d_output_sizes_vec;
            thrust::host_vector<nvcompStatus_t> h_statuses = d_statuses_vec;
            
            if (h_statuses[0] == nvcompSuccess) {
                result.compressed_size_bytes = h_output_sizes[0];
                result.success = true;
            } else {
                result.error_message = "LZ4 compression failed: " + std::to_string(h_statuses[0]);
            }
            break;
        }
        
        case nvcompAlgorithm::SNAPPY: {
            size_t temp_bytes;
            nvcompStatus_t status = nvcompBatchedSnappyCompressGetTempSizeAsync(
                1, input_size_bytes, nvcompBatchedSnappyCompressDefaultOpts,
                &temp_bytes, input_size_bytes);
            if (status != nvcompSuccess) {
                result.error_message = "Failed to get Snappy temp size: " + std::to_string(status);
                return result;
            }
            
            void* d_temp;
            checkCudaErrors(cudaMalloc(&d_temp, temp_bytes));
            
            status = nvcompBatchedSnappyCompressAsync(
                thrust::raw_pointer_cast(d_input_ptrs_vec.data()),
                thrust::raw_pointer_cast(d_input_sizes_vec.data()),
                input_size_bytes,
                1,
                d_temp,
                temp_bytes,
                thrust::raw_pointer_cast(d_output_ptrs_vec.data()),
                thrust::raw_pointer_cast(d_output_sizes_vec.data()),
                nvcompBatchedSnappyCompressDefaultOpts,
                thrust::raw_pointer_cast(d_statuses_vec.data()),
                stream);
            
            checkCudaErrors(cudaStreamSynchronize(stream));
            checkCudaErrors(cudaFree(d_temp));
            
            thrust::host_vector<size_t> h_output_sizes = d_output_sizes_vec;
            thrust::host_vector<nvcompStatus_t> h_statuses = d_statuses_vec;
            
            if (h_statuses[0] == nvcompSuccess) {
                result.compressed_size_bytes = h_output_sizes[0];
                result.success = true;
            } else {
                result.error_message = "Snappy compression failed: " + std::to_string(h_statuses[0]);
            }
            break;
        }
        
        case nvcompAlgorithm::GDEFLATE: {
            size_t temp_bytes;
            nvcompStatus_t status = nvcompBatchedGdeflateCompressGetTempSizeAsync(
                1, input_size_bytes, nvcompBatchedGdeflateCompressDefaultOpts,
                &temp_bytes, input_size_bytes);
            if (status != nvcompSuccess) {
                result.error_message = "Failed to get GDeflate temp size: " + std::to_string(status);
                return result;
            }
            
            void* d_temp;
            checkCudaErrors(cudaMalloc(&d_temp, temp_bytes));
            
            status = nvcompBatchedGdeflateCompressAsync(
                thrust::raw_pointer_cast(d_input_ptrs_vec.data()),
                thrust::raw_pointer_cast(d_input_sizes_vec.data()),
                input_size_bytes,
                1,
                d_temp,
                temp_bytes,
                thrust::raw_pointer_cast(d_output_ptrs_vec.data()),
                thrust::raw_pointer_cast(d_output_sizes_vec.data()),
                nvcompBatchedGdeflateCompressDefaultOpts,
                thrust::raw_pointer_cast(d_statuses_vec.data()),
                stream);
            
            checkCudaErrors(cudaStreamSynchronize(stream));
            checkCudaErrors(cudaFree(d_temp));
            
            thrust::host_vector<size_t> h_output_sizes = d_output_sizes_vec;
            thrust::host_vector<nvcompStatus_t> h_statuses = d_statuses_vec;
            
            if (h_statuses[0] == nvcompSuccess) {
                result.compressed_size_bytes = h_output_sizes[0];
                result.success = true;
            } else {
                result.error_message = "GDeflate compression failed: " + std::to_string(h_statuses[0]);
            }
            break;
        }
        
        default:
            result.error_message = "Algorithm " + nvcomp_algorithm_name(algorithm) + " not yet implemented";
            return result;
    }
    
    result.compression_time_ms = sw.ElapsedMs();
    return result;
}

nvcompDecompressionResult decompress_nvcomp(
    const uint8_t* d_compressed_data,
    size_t compressed_size_bytes,
    size_t original_size_bytes,
    nvcompAlgorithm algorithm,
    uint8_t* d_output_data)
{
    nvcompDecompressionResult result;
    result.success = false;
    result.decompressed_size_bytes = 0;
    
    StopWatch sw;
    cudaStream_t stream = cudaStreamDefault;
    
    // Create arrays for batched API (single chunk)
    thrust::device_vector<const void*> d_compressed_ptrs_vec(1);
    thrust::device_vector<size_t> d_compressed_sizes_vec(1);
    thrust::device_vector<void*> d_output_ptrs_vec(1);
    thrust::device_vector<size_t> d_output_buffer_sizes_vec(1);  // Buffer sizes (input)
    thrust::device_vector<size_t> d_output_chunk_sizes_vec(1);   // Actual decompressed sizes (output)
    thrust::device_vector<nvcompStatus_t> d_statuses_vec(1);
    
    d_compressed_ptrs_vec[0] = d_compressed_data;
    d_compressed_sizes_vec[0] = compressed_size_bytes;
    d_output_ptrs_vec[0] = d_output_data;
    d_output_buffer_sizes_vec[0] = original_size_bytes;
    
    switch (algorithm) {
        case nvcompAlgorithm::LZ4: {
            size_t temp_bytes;
            nvcompStatus_t status = nvcompBatchedLZ4DecompressGetTempSizeAsync(
                1, original_size_bytes, nvcompBatchedLZ4DecompressDefaultOpts,
                &temp_bytes, original_size_bytes);
            if (status != nvcompSuccess) {
                result.error_message = "Failed to get LZ4 decompress temp size: " + std::to_string(status);
                return result;
            }
            
            void* d_temp;
            checkCudaErrors(cudaMalloc(&d_temp, temp_bytes));
            
            status = nvcompBatchedLZ4DecompressAsync(
                thrust::raw_pointer_cast(d_compressed_ptrs_vec.data()),
                thrust::raw_pointer_cast(d_compressed_sizes_vec.data()),
                thrust::raw_pointer_cast(d_output_buffer_sizes_vec.data()),
                thrust::raw_pointer_cast(d_output_chunk_sizes_vec.data()),
                1,
                d_temp,
                temp_bytes,
                thrust::raw_pointer_cast(d_output_ptrs_vec.data()),
                nvcompBatchedLZ4DecompressDefaultOpts,
                thrust::raw_pointer_cast(d_statuses_vec.data()),
                stream);
            
            checkCudaErrors(cudaStreamSynchronize(stream));
            checkCudaErrors(cudaFree(d_temp));
            
            thrust::host_vector<nvcompStatus_t> h_statuses = d_statuses_vec;
            thrust::host_vector<size_t> h_output_chunk_sizes = d_output_chunk_sizes_vec;
            
            if (h_statuses[0] == nvcompSuccess) {
                result.decompressed_size_bytes = h_output_chunk_sizes[0];
                result.success = true;
            } else {
                result.error_message = "LZ4 decompression failed: " + std::to_string(h_statuses[0]);
            }
            break;
        }
        
        case nvcompAlgorithm::SNAPPY: {
            size_t temp_bytes;
            nvcompStatus_t status = nvcompBatchedSnappyDecompressGetTempSizeAsync(
                1, original_size_bytes, nvcompBatchedSnappyDecompressDefaultOpts,
                &temp_bytes, original_size_bytes);
            if (status != nvcompSuccess) {
                result.error_message = "Failed to get Snappy decompress temp size: " + std::to_string(status);
                return result;
            }
            
            void* d_temp;
            checkCudaErrors(cudaMalloc(&d_temp, temp_bytes));
            
            status = nvcompBatchedSnappyDecompressAsync(
                thrust::raw_pointer_cast(d_compressed_ptrs_vec.data()),
                thrust::raw_pointer_cast(d_compressed_sizes_vec.data()),
                thrust::raw_pointer_cast(d_output_buffer_sizes_vec.data()),
                thrust::raw_pointer_cast(d_output_chunk_sizes_vec.data()),
                1,
                d_temp,
                temp_bytes,
                thrust::raw_pointer_cast(d_output_ptrs_vec.data()),
                nvcompBatchedSnappyDecompressDefaultOpts,
                thrust::raw_pointer_cast(d_statuses_vec.data()),
                stream);
            
            checkCudaErrors(cudaStreamSynchronize(stream));
            checkCudaErrors(cudaFree(d_temp));
            
            thrust::host_vector<nvcompStatus_t> h_statuses = d_statuses_vec;
            thrust::host_vector<size_t> h_output_chunk_sizes = d_output_chunk_sizes_vec;
            
            if (h_statuses[0] == nvcompSuccess) {
                result.decompressed_size_bytes = h_output_chunk_sizes[0];
                result.success = true;
            } else {
                result.error_message = "Snappy decompression failed: " + std::to_string(h_statuses[0]);
            }
            break;
        }
        
        case nvcompAlgorithm::GDEFLATE: {
            size_t temp_bytes;
            nvcompStatus_t status = nvcompBatchedGdeflateDecompressGetTempSizeAsync(
                1, original_size_bytes, nvcompBatchedGdeflateDecompressDefaultOpts,
                &temp_bytes, original_size_bytes);
            if (status != nvcompSuccess) {
                result.error_message = "Failed to get GDeflate decompress temp size: " + std::to_string(status);
                return result;
            }
            
            void* d_temp;
            checkCudaErrors(cudaMalloc(&d_temp, temp_bytes));
            
            status = nvcompBatchedGdeflateDecompressAsync(
                thrust::raw_pointer_cast(d_compressed_ptrs_vec.data()),
                thrust::raw_pointer_cast(d_compressed_sizes_vec.data()),
                thrust::raw_pointer_cast(d_output_buffer_sizes_vec.data()),
                thrust::raw_pointer_cast(d_output_chunk_sizes_vec.data()),
                1,
                d_temp,
                temp_bytes,
                thrust::raw_pointer_cast(d_output_ptrs_vec.data()),
                nvcompBatchedGdeflateDecompressDefaultOpts,
                thrust::raw_pointer_cast(d_statuses_vec.data()),
                stream);
            
            checkCudaErrors(cudaStreamSynchronize(stream));
            checkCudaErrors(cudaFree(d_temp));
            
            thrust::host_vector<nvcompStatus_t> h_statuses = d_statuses_vec;
            thrust::host_vector<size_t> h_output_chunk_sizes = d_output_chunk_sizes_vec;
            
            if (h_statuses[0] == nvcompSuccess) {
                result.decompressed_size_bytes = h_output_chunk_sizes[0];
                result.success = true;
            } else {
                result.error_message = "GDeflate decompression failed: " + std::to_string(h_statuses[0]);
            }
            break;
        }
        
        default:
            result.error_message = "Algorithm " + nvcomp_algorithm_name(algorithm) + " not yet implemented";
            return result;
    }
    
    result.decompression_time_ms = sw.ElapsedMs();
    return result;
}
