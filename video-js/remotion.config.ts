import {Config} from '@remotion/cli/config';

Config.setEntryPoint('src/index.ts');
Config.setCodec('h264');
Config.setCrf(16);
Config.setVideoImageFormat('png'); // pas de JPEG intermédiaire : les trames des coupures restent nettes
Config.setOverwriteOutput(true);
