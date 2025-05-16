rng('shuffle');
metroRange = metropolisRange('ConstantVariance', 3, 0, 9, 10, 1);

maxTime = 100;

metroRange = metroRange.runHistogramToTimeT(maxTime);
display(metroRange.histogram);