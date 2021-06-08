#include <spatial_aligner/DataPreparationPipeline.h>
#include <iostream>

namespace feature_extraction {


DataPreparationPipeline::DataPreparationPipeline(fs::path dataPath, size_t nJobs) 
    : _semaphore(nJobs)
    , _dataPath(dataPath)
{
}


ExtractionResults DataPreparationPipeline::getResults()
{
    return _results;
}


std::pair<cv::Mat, std::vector<cv::KeyPoint>> DataPreparationPipeline::SIFTfeatureExtractor(std::future<cv::Mat> imgFuture)
{
    std::scoped_lock l(_semaphore);

    auto img = imgFuture.get();
    cv::Mat grayscale;
    // resize and convert to grayscale
    if(img.rows > 1080)
    {
        cv::resize(img, img, cv::Size(1080, 720));
    }
    cv::cvtColor(img, grayscale, cv::COLOR_BGR2GRAY);

    // extract keypoitns
    std::vector<cv::KeyPoint> keypoints;
    try 
    {
        cv::Ptr<cv::SIFT> detector = cv::SiftFeatureDetector::create(); 
        detector->detect(img, keypoints);
    }
    catch(std::exception& e)
    {
        std::cout << "Keypoints detection exception ==> " << e.what() << std::endl;
    }   

    return {img, keypoints};
}


std::tuple<cv::Mat, std::vector<cv::KeyPoint>, cv::Mat> 
DataPreparationPipeline::SIFTDescriptorExtractor(std::future<std::pair<cv::Mat, std::vector<cv::KeyPoint>>> imgWithKeypointsFuture)
{
    auto imgWithKeypoints = imgWithKeypointsFuture.get();

    auto& img = imgWithKeypoints.first;
    auto& keypoints = imgWithKeypoints.second;

    // extract descriptors
    cv::Mat descriptors;
    try 
    {
        cv::Ptr<cv::SiftDescriptorExtractor> extractor = cv::SiftDescriptorExtractor::create();
        extractor->compute(img, keypoints, descriptors);
    }
    catch(std::exception& e)
    {
        std::cout << "Descriptors extraction exception ===> " << e.what() << std::endl;
    }

    return {img, keypoints, descriptors};
}


cv::Mat DataPreparationPipeline::loadImage(std::future<std::string> fileNameFuture)
{
    std::scoped_lock l(_semaphore);

    auto fileName = fileNameFuture.get();

    auto img = cv::imread(fileName);

    return img;
}


void DataPreparationPipeline::run() 
{
    std::unordered_map<std::string, std::future<std::tuple<cv::Mat, std::vector<cv::KeyPoint>, cv::Mat>>> futures;

    for(auto& entry : fs::directory_iterator(_dataPath))
    {
        std::string imgPath(entry.path().u8string());

        // wrap the file name into a future to have future inputs for each "stage"
        std::promise<std::string> wrapFileName;
        std::future<std::string> fnFuture = wrapFileName.get_future();
        wrapFileName.set_value(imgPath);

        // ============= STAGE 1: LOAD IMAGE FROM DISK ============================================
        auto imgFuture = std::async(std::launch::async, &DataPreparationPipeline::loadImage, this, std::move(fnFuture));
        
        // ============= STAGE 2: EXTRACT KEYPOINTS FROM LOADED IMAGE ==============================
        auto imgWithKeypoints = std::async(std::launch::async, &DataPreparationPipeline::SIFTfeatureExtractor, this, std::move(imgFuture));

        // ============= STAGE 3: EXTRACT DESCRIPTORS FROM LOADED IMAGE AND KEYPOINTS ==============
        auto imgKeypointsAndDescriptors = std::async(std::launch::async, &DataPreparationPipeline::SIFTDescriptorExtractor, this, std::move(imgWithKeypoints));


        futures.emplace(imgPath, std::move(imgKeypointsAndDescriptors));
    }

    for(auto& it : futures)
    {
        _results.emplace(it.first, it.second.get());
    }
}

} // namespace spatial_aligner