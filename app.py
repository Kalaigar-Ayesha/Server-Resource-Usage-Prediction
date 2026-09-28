from flask import Flask, request, jsonify, render_template
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            data = request.json
        else:
            data = request.form
            
        custom_data = CustomData(
            cpu_usage=float(data.get('cpu_usage', 0)),
            memory_usage=float(data.get('memory_usage', 0)),
            disk_usage=float(data.get('disk_usage', 0)),
            network_in=float(data.get('network_in', 0)),
            network_out=float(data.get('network_out', 0)),
            request_count=float(data.get('request_count', 0)),
            response_time=float(data.get('response_time', 0)),
            timestamp=data.get('timestamp')
        )
        
        features = custom_data.get_data_as_dict()
        pipeline = PredictPipeline()
        prediction = pipeline.predict(features)
        
        if request.is_json:
            return jsonify({
                'status': 'success',
                'future_cpu_usage': float(prediction)
            })
        else:
            return render_template('index.html', results=round(float(prediction), 2), form_data=data)
            
    except Exception as e:
        if request.is_json:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 400
        else:
            return render_template('index.html', error=str(e), form_data=request.form)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
