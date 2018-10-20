# -*- mode: ruby -*-
# vi: set ft=ruby

VAGRANTFILE_API_VERSION = "2"

Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|
	config.vm.box = "xenial64"
	config.vm.hostname = "test-vagrant"
	config.vm.box_url = "https://cloud-images.ubuntu.com/xenial/current/xenial-server-cloudimg-amd64-vagrant.box"
	# config.vm.network :private_network, ip: "192.168.11.11"
	config.vm.network "forwarded_port", guest: 8888, host: 8888
	config.vm.synced_folder "src/", "/vagrant/src", id: "code"
	config.vm.synced_folder "D:/vagrant-test", "/vagrant/imgs", id: "images"
	config.vm.provider :virtualbox do |v|
		v.customize ["modifyvm", :id, "--memory", 8191]
	end
end