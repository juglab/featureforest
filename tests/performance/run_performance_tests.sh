conda activate featureforest

models=($(python -c "from featureforest.models import get_available_models; print(' '.join(get_available_models()))"))

sizes=(
    "512 512"
    "1024 1024"
    "2048 2048"
    "4096 4096"
    "6144 6144"
    "8192 8192"
)

# Run each experiment configuration separately
for model in "${models[@]}"; do
    for size in "${sizes[@]}"; do
        read height width <<< "$size"
        echo "Running experiment: Model=$model, Size=${height}x${width}"
        
        # Run the experiment
        python measure_runtime_stats.py "$model" "$height" "$width"
        
        # Sleep for a few seconds to ensure GPU memory is fully cleared
        sleep 15
    done
done

echo "All experiments completed" 