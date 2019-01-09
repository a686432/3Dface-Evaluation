import 


class MeshopTool(object):
    def __init__(self, mesh_name = None, meshlist = None):
        self.mesh_name = mesh_name
        self.mesh_



    def _rotate(self,mesh_vertices,theta):
        # rotation_matrix=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
        # mesh_vertices[:,0:2]=np.dot(mesh_vertices[:,0:2],rotation_matrix)
        #mesh_vertices[:,1] = -mesh_vertices[:,1]
        # theta=0.11
        rotation_matrix=np.array([[math.cos(theta),0, -math.sin(theta)],[0,1,0],[-math.sin(theta),0,math.cos(theta)]])
        rotation_matrix=np.array([[math.cos(theta),0, -math.sin(theta)],[0,1,0],[-math.sin(theta),0,math.cos(theta)]])
        # mesh_vertices[:,0:3]=np.dot(mesh_vertices[:,0:3],rotation_matrix)
        # #print(mesh_vertices)
        return mesh_vertices


    def read_mesh(self,file_name):
        with open(file_name, "r") as f:
            lines=f.readlines()
            vertices=[]
            faces=[]
            for line in lines:
                words = line.split(" ")
                if words[0]=="v":
                    ver=np.zeros(5)
                    ver[:]=float(words[1]),float(words[2]),float(words[3]),1,-1
                    vertices.append(ver)
                if words[0]=="f":
                    face=np.zeros(4,dtype=int)
                    face[:]=int(words[1]),int(words[2]),int(words[3]),1
                    faces.append(face)
            vertices=np.array(vertices)
            faces=np.array(faces)
        return vertices,faces

    def save_mesh(self,mesh_vertices, mesh_face, output_name):
        num_vertices = mesh_vertices.shape[0]
        num_face = mesh_face.shape[0]
        f = open(output_name, "w")
        f.write("# {} vertices, {} faces\n".format(num_vertices, num_face))
        for i in range(num_vertices):
            f.write("v {} {} {}\n".format(mesh_vertices[i][0], mesh_vertices[i][1], mesh_vertices[i][2]))
        for i in range(num_face):
            f.write("f {} {} {}\n".format(mesh_face[i][0], mesh_face[i][1], mesh_face[i][2]))
        f.close()


    