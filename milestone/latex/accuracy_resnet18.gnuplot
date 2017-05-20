set terminal png
set output "accuracy_resnet18.png"
set   autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "Top-1 Accuracy vs Epoch"
set xlabel "Epoch"
set ylabel "Top-1 Accuracy"
plot    "accuracy_resnet18.data" using 1:2 title 'Training' with linespoints , \
          "accuracy_resnet18.data" using 1:3 title 'Validation' with linespoints
