The pvlib library itself is 400MB hence we cannot make a layer and add it to a lambda.

We can execute docker containers as lambdas

Instructions to create a docker image for lambda( aws-lambda-container-images ) :

https://towardsdatascience.com/building-aws-lambda-container-images-5c4a9a15e8a2

Instructions to run locally :

docker build -t pvforecast .

docker run -p 9000:8080 pvforecast

from another terminal curl request

curl --request POST --url http://localhost:9000/2015-03-31/functions/function/invocations --header 'Content-Type: application/json' --data '{"Latitude":"14.97","Longitude":"77.59"}'

I have deployed the function on our aws account here is the api gateway request

https://3mpqory1se.execute-api.ap-south-1.amazonaws.com/prod/pvforecast?Latitude=14.97&Longitude=77.59

Things needed to be done :

1. The unit of the ac/dc output
2. provide historic weather predictions to get past 1 years power output
3. configurations for the type of solar cells need to be corrected to actual values in practice
