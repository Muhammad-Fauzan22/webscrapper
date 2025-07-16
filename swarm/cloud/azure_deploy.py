#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AZURE DEPLOYMENT MANAGER
Version: 2.0.0
Created: 2025-07-17
Author: Cloud Operations Team
"""

import os
import logging
from azure.identity import DefaultAzureCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.containerinstance.models import (
    ContainerGroup,
    Container,
    ResourceRequests,
    ResourceRequirements,
    ContainerGroupRestartPolicy,
    OperatingSystemTypes,
    IpAddress,
    Port,
    ContainerGroupNetworkProtocol,
    EnvironmentVariable
)

logger = logging.getLogger("AzureDeployer")

class AzureDeployer:
    def __init__(self):
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        self.resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        self.credential = DefaultAzureCredential()
        self.client = ContainerInstanceManagementClient(
            self.credential,
            self.subscription_id
        )
    
    async def deploy_container(self, image_name: str, container_name: str) -> dict:
        """Deploy a new container instance"""
        try:
            # Configure container resources
            container_resource_requests = ResourceRequests(
                cpu=1.0,
                memory_in_gb=1.5
            )
            container_resource_requirements = ResourceRequirements(
                requests=container_resource_requests
            )
            
            # Create container configuration
            container = Container(
                name=container_name,
                image=image_name,
                resources=container_resource_requirements,
                ports=[Port(port=80)],
                environment_variables=self._get_env_variables()
            )
            
            # Configure IP address
            ip_address = IpAddress(
                ports=[Port(protocol=ContainerGroupNetworkProtocol.tcp, port=80)],
                type="Public"
            )
            
            # Create container group
            container_group = ContainerGroup(
                location="eastus",
                containers=[container],
                os_type=OperatingSystemTypes.linux,
                ip_address=ip_address,
                restart_policy=ContainerGroupRestartPolicy.always
            )
            
            # Deploy container group
            deployment = self.client.container_groups.begin_create_or_update(
                resource_group_name=self.resource_group,
                container_group_name=container_name,
                container_group=container_group
            ).result()
            
            logger.info(f"Container deployed: {deployment.name}")
            return {
                "status": "success",
                "ip_address": deployment.ip_address.ip,
                "fqdn": deployment.ip_address.fqdn
            }
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def restart_container(self, container_name: str) -> bool:
        """Restart a running container instance"""
        try:
            self.client.container_groups.restart(
                resource_group_name=self.resource_group,
                container_group_name=container_name
            )
            logger.info(f"Container restarted: {container_name}")
            return True
        except Exception as e:
            logger.error(f"Restart failed: {str(e)}")
            return False
    
    async def update_container(self, container_name: str, new_image: str) -> bool:
        """Update container with new image version"""
        try:
            # Get current container group
            cg = self.client.container_groups.get(
                resource_group_name=self.resource_group,
                container_group_name=container_name
            )
            
            # Update container image
            cg.containers[0].image = new_image
            
            # Apply update
            self.client.container_groups.begin_create_or_update(
                resource_group_name=self.resource_group,
                container_group_name=container_name,
                container_group=cg
            ).result()
            
            logger.info(f"Container updated with image: {new_image}")
            return True
        except Exception as e:
            logger.error(f"Update failed: {str(e)}")
            return False
    
    def _get_env_variables(self) -> list:
        """Get required environment variables"""
        return [
            EnvironmentVariable(name="MONGO_URI", value=os.getenv("MONGO_URI")),
            EnvironmentVariable(name="GDRIVE_FOLDER_ID", value=os.getenv("GDRIVE_FOLDER_ID")),
            EnvironmentVariable(name="HF_TOKEN", value=os.getenv("HF_TOKEN")),
            EnvironmentVariable(name="SCRAPEOPS_API_KEY", value=os.getenv("SCRAPEOPS_API_KEY")),
            EnvironmentVariable(name="DEEPSEEK_KEY", value=os.getenv("DEEPSEEK_KEY")),
            EnvironmentVariable(name="PERPLEXITY_KEY", value=os.getenv("PERPLEXITY_KEY")),
            EnvironmentVariable(name="CLAUDE_KEY", value=os.getenv("CLAUDE_KEY")),
            EnvironmentVariable(name="CYPHER_KEY", value=os.getenv("CYPHER_KEY")),
            EnvironmentVariable(name="GEMMA_KEY", value=os.getenv("GEMMA_KEY")),
            EnvironmentVariable(name="TOKEN_BUDGET", value=os.getenv("TOKEN_BUDGET", "100000")),
        ]
