from flask_restful import Resource

class Hello(Resource):
    def get(self):
        return {"message": "Hello"}
    
    def post(self):
        return {"message": "Hello"}
    
