docker run --name test-mysql -v db:/var/lib/mysql  --network host -e MYSQL_ROOT_PASSWORD=root  -e MYSQL_DATABASE=marvin -dit mysql:latest --default-authentication-plugin=mysql_native_password
