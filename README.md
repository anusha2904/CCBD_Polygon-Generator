# CCBD_Polygon-Generator
Install the modules mentioned in the module_file or run this command:
$ pip install -r requirements.txt

Run the python file in the terminal:
python3 polygon_generator.py

This project aims to generate random polygon ranging from 10 to 500 vertices with base polygons ranging from triangle to octagon. The polygons are saved in a wkt format in a .csv file.  
![image](https://user-images.githubusercontent.com/65866016/117934939-cdeb3000-b320-11eb-8e29-50e53f4481ce.png)

The time taken to generate the file and the size of file is
![image](https://user-images.githubusercontent.com/65866016/117932519-13f2c480-b31e-11eb-8ff1-e7d8957dbbcb.png)

The polygons generated in the .csv file are visualised using QGIS, an open-source cross-platform desktop geographic information system application.
![image](https://user-images.githubusercontent.com/65866016/117930430-99c14080-b31b-11eb-8a7b-a700749fc0dc.png)

The polygon is later used to generate a Voronoi map which represents spatial bodies in real-time maps.
![image](https://user-images.githubusercontent.com/65866016/117930630-d8ef9180-b31b-11eb-9a04-e54c44a6a61c.png)

The image of the map is processed to find the distribution of these spatial bodies on the map using PI chart
![image](https://user-images.githubusercontent.com/65866016/117930599-cffec000-b31b-11eb-8687-14376385fdaa.png)


