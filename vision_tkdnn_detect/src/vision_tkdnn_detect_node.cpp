/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ********************
 *  v1.0: amc-nu (abrahammonrroy@yahoo.com)
 *
 *  Created on: April 4th, 2018
 */

#include "vision_tkdnn_detect.h"

#include <signal.h>

tkdnn::BoundingBoxDetector *app;

void bbdSignalHandler(int sig)
{
  // Do some custom action.
  // For example, publish a stop message to some other nodes.

  // All the default sigint handler does is call shutdown()
  app->printStats();

  ros::shutdown();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vision_tkdnn_detect", ros::init_options::NoSigintHandler);

    signal(SIGINT, bbdSignalHandler);
    app = new tkdnn::BoundingBoxDetector();

    app->Run();

    delete app;

    return 0;
}
