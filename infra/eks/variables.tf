variable "region" {
  default = "ap-southeast-1"
}

variable "cluster_name" {
  default = "eks-ap-southeast-1"
}

variable "node_instance_type" {
  default = "t3.medium"
}

variable "desired_nodes" {
  default = 1
}