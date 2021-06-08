#include <filesystem>  
#include <future>
#include <list>
#include <vector>
#include <memory>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>


namespace fs = std::filesystem;

namespace feature_extraction {

// Control the number of threads that will be lunched
class Semaphore
{
public:
    explicit Semaphore(size_t jobs)
        : _jobs(jobs)
    {
    }

    void lock()
    {
        std::unique_lock<std::mutex> l(_mtx);
        _cv.wait(l, [this] {
            return _jobs > 0;
        });

        -- _jobs;
    }

    void unlock()
    {
        std::unique_lock<std::mutex> l(_mtx);
        ++ _jobs;

        _cv.notify_one();
    }

private:
    size_t _jobs;

    std::mutex _mtx;
    std::condition_variable _cv;
};

// key - filenap, value - image, keypoints and descriptors
typedef std::unordered_map<std::string, std::tuple<cv::Mat, std::vector<cv::KeyPoint>, cv::Mat>> ExtractionResults;

/**
 * Responsible to load images from a directory and to extract features(using SIFT) from them.
*/
class DataPreparationPipeline
{
public:
    explicit DataPreparationPipeline(fs::path dataPath, size_t nJobs);
    
    void run();

    // the key is the path to the file and the value is a tuple of image + extracted keypoints + descriptors
    ExtractionResults getResults();

private:
    cv::Mat loadImage(std::future<std::string> fileNameFuture);

    std::pair<cv::Mat, std::vector<cv::KeyPoint>> SIFTfeatureExtractor(std::future<cv::Mat> imgFuture);

    std::tuple<cv::Mat, std::vector<cv::KeyPoint>, cv::Mat>
    SIFTDescriptorExtractor(std::future<std::pair<cv::Mat, std::vector<cv::KeyPoint>>> imgWithKeypointsFuture);


    Semaphore _semaphore;
    fs::path _dataPath;
    ExtractionResults _results;
};

} // namespace spatial_aligner