#include <iostream>
#include <string>
#include <spatial_aligner/DataPreparationPipeline.h>


void visualizeResults(const panorama_generation::feature_extraction::ExtractionResults& results)
{
    for(auto& it : results)
    {
        std::cout << it.first << " have descriptor : " << !std::get<2>(it.second).empty() << std::endl;
        
        auto img = std::get<0>(it.second).clone();
        cv::drawKeypoints(img, std::get<1>(it.second), img);

        cv::namedWindow("image", cv::WINDOW_NORMAL);;
        cv::imshow("image", img);
        cv::waitKey(0);
    }
}


int main()
{
    auto buildPath = fs::current_path();

    fs::path dataPath = buildPath / "data";

    auto pipeline = std::make_shared<panorama_generation::feature_extraction::DataPreparationPipeline>(dataPath, /*nThreads*/ 8);

    auto start = std::chrono::system_clock::now();
    {
        pipeline->run();
    }
    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double> >(end - start);
    
    std::cout << "Runtime ==========> " << elapsed.count() << std::endl;

    panorama_generation::feature_extraction::ExtractionResults results = pipeline->getResults();
    visualizeResults(results);


    return 0;
}
