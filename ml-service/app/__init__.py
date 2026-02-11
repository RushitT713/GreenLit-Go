from flask import Flask
from flask_cors import CORS
import os
from dotenv import load_dotenv

load_dotenv()

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Register routes
    from app.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    @app.route('/health')
    def health():
        return {'status': 'ok', 'service': 'GreenLit Go ML Service'}
    
    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
